## Importing all necessary libraries 

import pandas as pd

import numpy as np



import seaborn as sns



import matplotlib.pyplot as plt

%matplotlib inline



import tensorflow as tf



from tensorflow.keras import layers

from tensorflow.keras.models import Model

from tensorflow.keras import metrics

from tensorflow.keras import backend as K
# Reading the folder architecture of Kaggle to get the dataset path.

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Reading the Datasets.

mnist_train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

mnist_test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")
# Printing the dimensions/shape of the given data

print(mnist_train.shape, mnist_test.shape)
# Preliminary analysis

mnist_train.head()
mnist_train.describe()
mnist_train.isna().any().any()
# dividing the data into the input and output features to train make the model learn based on what to take in and what to throw out.

mnist_train_data = mnist_train.loc[:, "pixel0":]

mnist_train_label = mnist_train.loc[:, "label"]



# Normalizing the images array to be in the range of 0-1 by dividing them by the max possible value. 

# Here is it 255 as we have 255 value range for pixels of an image. 

mnist_train_data = mnist_train_data/255.0

mnist_test = mnist_test/255.0
# Let's make some beautiful plots.

digit_array = mnist_train.loc[3, "pixel0":]

arr = np.array(digit_array) 



#.reshape(a, (28,28))

image_array = np.reshape(arr, (28,28))



digit_img = plt.imshow(image_array, cmap=plt.cm.binary)

plt.colorbar(digit_img)

print("IMAGE LABEL: {}".format(mnist_train.loc[3, "label"]))
# Let's make some beautiful plots.

digit_array = mnist_train.loc[4, "pixel0":]

arr = np.array(digit_array) 



#.reshape(a, (28,28))

image_array = np.reshape(arr, (28,28))



digit_img = plt.imshow(image_array, cmap=plt.cm.binary)

plt.colorbar(digit_img)

print("IMAGE LABEL: {}".format(mnist_train.loc[4, "label"]))
# Let's build a count plot to see the count of all the labels.

sns.countplot(mnist_train.label)

print(list(mnist_train.label.value_counts().sort_index()))
# Converting dataframe into arrays

mnist_train_data = np.array(mnist_train_data)

mnist_train_label = np.array(mnist_train_label)
# Reshaping the input shapes to get it in the shape which the model expects to recieve later.

mnist_train_data = mnist_train_data.reshape(mnist_train_data.shape[0], 28, 28, 1)

print(mnist_train_data.shape, mnist_train_label.shape)
# But first import some cool libraries before getting our hands dirty!! 

# TensorFlow is Google's open source AI framework and we are using is here to build model.

# Keras is built on top of Tensorflow and gives us

# NO MORE GEEKY STUFF, Know more about them here:  https://www.tensorflow.org     https://keras.io



from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout, Lambda, Flatten, BatchNormalization

from tensorflow.keras.layers import Conv2D, MaxPool2D, AvgPool2D

from tensorflow.keras.optimizers import Adadelta

from keras.utils.np_utils import to_categorical

from tensorflow.keras.preprocessing.image import ImageDataGenerator

### When the accuracy or loss starts to plateau during training we can implement the following callbacks 

#### to lower the learning rate and hence make smaller steps as it gets closer to the global optimum

from tensorflow.keras.callbacks import ReduceLROnPlateau

from tensorflow.keras.callbacks import LearningRateScheduler
# Encoding the labels and making them as the class value and finally converting them as categorical values.

nclasses = mnist_train_label.max() - mnist_train_label.min() + 1

mnist_train_label = to_categorical(mnist_train_label, num_classes = nclasses)

print("Shape of y_train after encoding: ", mnist_train_label.shape)
# This function builds the CNN Necessary for recognition, detailed explanation in the comments



def build_model(input_shape=(28, 28, 1)):

    model = Sequential()     ## We need a sequential model obviously (don't require bidirectional, etc)

    model.add(Conv2D(32, kernel_size = 3, activation='relu', input_shape = input_shape)) # First 2D Convolutional layer

    model.add(BatchNormalization()) # Activation is Rectified Linear Unit of ReLU for all layers

    model.add(Conv2D(32, kernel_size = 3, activation='relu')) # Batch Normalization is used along with Dropout

    model.add(BatchNormalization())

    model.add(Conv2D(32, kernel_size = 5, strides=2, padding='same', activation='relu'))

    model.add(BatchNormalization())

    model.add(Dropout(0.4))

    ## Dropout Regularization of 0.4 in order to avoid overfitting

    model.add(Conv2D(64, kernel_size = 3, activation='relu'))

    model.add(BatchNormalization())

    model.add(Conv2D(64, kernel_size = 3, activation='relu'))

    model.add(BatchNormalization())

    model.add(Conv2D(64, kernel_size = 5, strides=2, padding='same', activation='relu'))

    model.add(BatchNormalization())

    model.add(Dropout(0.4))



    model.add(Conv2D(128, kernel_size = 4, activation='relu'))

    model.add(BatchNormalization())

    model.add(Flatten())

    model.add(Dropout(0.4))

    model.add(Dense(10, activation='softmax')) ## Softmax activation used as this is a multiclass classification task

    return model    ## The number of units is 10 as there are 10 different classes of digits
def compile_model(model, optimizer='adam', loss='categorical_crossentropy'):

    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"]) # using adam optimization, RMSProp works fine too

    ## Categorical crossentropy is used as the multiclass loss

    

def train_model(model, train, test, epochs, split):

    history = model.fit(train, test, shuffle=True, epochs=epochs, validation_split=split)

    return history ## Data is shuffled during training to avoid inherent bias to the sequence of occurence of an image
# Training the model using the above function built to build, compile and train the model

cnn_model = build_model((28, 28, 1)) ## The input is an image odf size 28 X 28

compile_model(cnn_model, 'adam', 'categorical_crossentropy')



# train the model for as many epochs as you want but I found training it above 100 will not help us and eventually 

## increase overfitting.

model_history = train_model(cnn_model, mnist_train_data, mnist_train_label, 50, 0.2)
def plot_model_performance(metric, validations_metric):

    plt.plot(model_history.history[metric],label = str('Training ' + metric))

    plt.plot(model_history.history[validations_metric],label = str('Validation ' + metric))

    plt.legend()
### Plotting the accuracy's

plot_model_performance('accuracy', 'val_accuracy')
## Plotting the loss

plot_model_performance('loss', 'val_loss')
# reshaping the test arrays as we did to train images above somewhere.

mnist_test_arr = np.array(mnist_test)

mnist_test_arr = mnist_test_arr.reshape(mnist_test_arr.shape[0], 28, 28, 1)

print(mnist_test_arr.shape)
# Now, since the model is trained, it's time to find the results for the unseen test images.

predictions = cnn_model.predict(mnist_test_arr)
# Finally, making the final submissions

predictions_test = []



for i in predictions:

    predictions_test.append(np.argmax(i))
## Submitting in the required format

submission =  pd.DataFrame({

        "ImageId": mnist_test.index+1,

        "Label": predictions_test

    })



submission.to_csv('submission.csv', index=False)
submission