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
# Reading the Train and Test Datasets.
fish_train = pd.read_csv("/kaggle/input/fishes/train.csv")
fish_test = pd.read_csv("/kaggle/input/fishes/test.csv")
# Let's see the shape of the train and test data
print(fish_train.shape, fish_test.shape)
# Looking at a few rows from the data isn't a bad idea.
fish_train.head()
# and yeah, here you will see the basic statistical insights of the numerical features of train data.
fish_train.describe()
fish_train.isna().any().any()
# dividing the data into the input and output features to train make the model learn based on what to take in and what to throw out.
fish_train_data = fish_train.loc[:, "#px0":]
fish_train_label = fish_train.loc[:, "Label"]

# Notmailzing the images array to be in the range of 0-1 by dividing them by the max possible value. 
# Here is it 255 as we have 255 value range for pixels of an image. 
fish_train_data = fish_train_data/255.0
fish_test = fish_test/255.0
# Let's make some beautiful plots.
digit_array = fish_train.loc[3, "#px0":]
arr = np.array(digit_array) 

#.reshape(a, (28,28))
image_array = np.reshape(arr, (28,28))

digit_img = plt.imshow(image_array, cmap=plt.cm.binary)
plt.colorbar(digit_img)
print("IMAGE LABEL: {}".format(fish_train.loc[3, "Label"]))
# Let's build a count plot to see the count of all the labels.
sns.countplot(fish_train.Label)
print(list(fish_train.Label.value_counts().sort_index()))
# Converting dataframe into arrays
fish_train_data = np.array(fish_train_data)
fish_train_label = np.array(fish_train_label)
# Reshaping the input shapes to get it in the shape which the model expects to recieve later.
fish_train_data = fish_train_data.reshape(fish_train_data.shape[0], 28, 28, 1)
print(fish_train_data.shape, fish_train_label.shape)
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
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import LearningRateScheduler
# Encoding the labels and making them as the class value and finally converting them as categorical values.
nclas = fish_train_label.max() - fish_train_label.min() + 1
print(nclas)
fish_train_label = to_categorical(fish_train_label, num_classes = nclas)
print("Shape of ytrain after encoding: ", fish_train_label.shape)
# Warning!!! Here comes the beast!!!

def build_model(input_shape=(28, 28, 1)):
    model = Sequential()
    model.add(Conv2D(32, kernel_size = 3, activation='relu', input_shape = input_shape))
    model.add(BatchNormalization())
    model.add(Conv2D(32, kernel_size = 3, activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(32, kernel_size = 5, strides=2, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

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
    model.add(Dense(10, activation='softmax'))
    return model

    
def compile_model(model, optimizer='adam', loss='categorical_crossentropy'):
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
    
    
def train_model(model, train, test, epochs, split):
    history = model.fit(train, test, shuffle=True, epochs=epochs, validation_split=split)
    return history
batch_size=10
epochs=10

from keras.callbacks import ReduceLROnPlateau
red_lr= ReduceLROnPlateau(monitor='val_acc',patience=3,verbose=1,factor=0.1)
print(fish_train_data.shape)
fish_train_label.shape
# Training the model using the above function built to build, compile and train the model
cnn_model = build_model((28, 28, 1))
compile_model(cnn_model, 'adam', 'categorical_crossentropy')

# train the model for as many epochs as you want but I found training it above 100 will not help us and eventually increase overfitting.
model_history = train_model(cnn_model, fish_train_data, fish_train_label, 5, 0.2)

def plot_model_performance(metric, validations_metric):
    plt.plot(model_history.history[metric],label = str('Training ' + metric))
    plt.plot(model_history.history[validations_metric],label = str('Validation ' + metric))
    plt.legend()
plot_model_performance('accuracy', 'val_accuracy')
plot_model_performance('loss', 'val_loss')
# reshaping the test arrays as we did to train images above somewhere.
fish_test_arr = np.array(fish_test)
fish_test_arr = fish_test_arr.reshape(fish_test_arr.shape[0], 28, 28, 1)
print(fish_test_arr.shape)
# Now, since the model is trained, it's time to find the results for the unseen test images.
predictions = cnn_model.predict(fish_test_arr)
# Finally, making the final submissions assuming that we have to submit it in any comptition. P)
predictions_test = []

for i in predictions:
    predictions_test.append(np.argmax(i))
submission =  pd.DataFrame({
        "ImageId": fish_test.index+1,
        "Label": predictions_test
    })

submission.to_csv('my_submission.csv', index=False)
submission.head()