**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./examples/model_summary.png "Model Summary"
[image2]: ./examples/final_model.png "Final Model"

---

### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup.md summarizing the results
* project.ipynd generate some diagram which I used to describe model

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing

```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I have tried LeNet at the very beginning, because I have use it before in Project2 and familiar with it. But this network cannot provide a good model. Then I changed to Nvidia's network. And that network after trained 3 epochs, produced a good enough model.

Here is the model summary:

![Model Summary][image1]

#### 2. Attempts to reduce overfitting in the model

I didn't use Dropout or Pooling to this model. Because I used the Nvidia's network, and that network architecture can convergence pretty well with after only 3 epochs.  

And the model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 144).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of the images from 3 camera(center, left, right). The codes of them are in `def combineImages(center, left, right, measurement, correction)` function in `model.py`

For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

I used the model architecture from NVidia Autonomous Car Group.

The only modification was to add a new layer at the end to have a single output as it was required: `model.add(Dense(1))`.

The car did its first complete track, but it's not perfect. There was a place in the track where it passes over the "dashed" line.

So I think we need more training data. In order to do that, I augmented the data by adding the same image flipped with a negative angle.

``` python
# Flipping, add more data
images.append(cv2.flip(image, 1))
angles.append(measurement * -1.0)
```

In addition to that, the left and right camera images where introduced with a correction factor on the angle to help the car go back to the lane(line 53 to 66 in `model.py`):

``` python
def combineImages(center, left, right, measurement, correction):
  """
  Combine the image paths from `center`, `left` and `right` using the correction factor `correction`
  Returns ([imagePaths], [measurements])
  """
  imagePaths = []
  imagePaths.extend(center)
  imagePaths.extend(left)
  imagePaths.extend(right)
  measurements = []
  measurements.extend(measurement)
  measurements.extend([x + correction for x in measurement])
  measurements.extend([x - correction for x in measurement])
  return (imagePaths, measurements)
```

#### 2. Final Model Architecture

The final neural network architecture is shown bellow

![Final Model][image2]

#### 3. Creation of the Training Set & Training Process

I only use the Udacity Data set.
And as I describe above, use all image from 3 camera(center, left, right).
And shuffled them randomly.
Also split the data set into training data and validation with 8:2 ratio.
Apart from that, I also augmented the data by flipping the image.
With all this technique, the Nvidia model produced a pretty well model.
