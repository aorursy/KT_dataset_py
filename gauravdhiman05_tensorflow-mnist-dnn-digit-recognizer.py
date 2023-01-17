# load all require packages.

import pandas as pd

import numpy as np

import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Flatten, Dense

import matplotlib.pyplot as plt

%matplotlib inline
# Load training data as dataframe

train = pd.read_csv("../input/mnist-in-csv/mnist_train.csv")

# first column is label, excluding that get the bitmap of image. One row in DF represents image of one hand-written number.

# also normalize the data by dividing the values by 255 (max RGB value)

train_image = train.iloc[:,1:]/255

train_image.head()
# load first column as training label

train_label = train.label

train_label.head()
#  load testing data to evaluate the model later.

test = pd.read_csv("../input/mnist-in-csv/mnist_test.csv")

test_label = test.label

test_label.head() # shows what are first five digits in test dataset.
#  separate the image butmap from label for testing dataset.

test_image = test.drop(columns='label')

test_image = test_image.values.reshape(test_image.shape[0], 28, 28)

test_image = tf.convert_to_tensor(test_image)

test_image[:,0] # bitmap of first image in test dataset.
#  method to show given number images.

def show_img(img_vector_arr, label_tensor):

    for i in range(len(img_vector_arr)):

        plt.subplot(290 + (i+1))

        plt.imshow(img_vector_arr[i],cmap=plt.get_cmap('gray'))

        if label_tensor is not None:

            plt.title(int(label_tensor[i]))
#  just show initial few number images and label to see the kind of data we have in training dataset.

train_image = train_image.values.reshape(train_image.shape[0], 28, 28)

train_image = tf.convert_to_tensor(train_image)

show_img(train_image[0:5], train_label)
# Add hidden layer of RELU neuron

model = Sequential()

model.add(Flatten(input_shape=(28,28)))

model.add(Dense(units=128, activation='relu'))

model.add(Dense(units=10, activation='softmax'))
# compile the model with right optimizer, loss function and metric to optimize.

model.compile(

    optimizer=tf.keras.optimizers.Adam(),

    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 

    metrics=['accuracy']

)
# train the model with training dataset.

model.fit(train_image, train_label, epochs=10)
#  evaluate how well the model is predicting with test dataset which is not yet seen by model. Calculate accuracy of model.

test_loss, test_acc = model.evaluate(test_image, test_label)

print('Accuracy of model is: {}%'.format(round(test_acc*100, 2)))

# load the totally new set of images for perdiction.

new_images = pd.read_csv("../input/digit-recognizer/test.csv")

new_images.head()
# use the model to predict what number is in images.

new_images = new_images.values.reshape(len(new_images.pixel0), 28, 28) # convert from 2D to 3D array

prediction = model.predict(new_images)

prediction = pd.DataFrame(prediction) # convert back the numpy array to pandas DF, for printing

prediction.head()
prediction = prediction.values.reshape(prediction.shape)

predicted_num={}

for i in range(0, len(prediction)):

    predicted_num[i] = [np.argmax(prediction[i])] # convert one-hot binary vector to integer array.

predicted_num = pd.DataFrame(predicted_num) # convert integer array to pandas DF, for printing.

predicted_num.head()
# show the predicted images along with predicted label on the top of the images.

# see all numbers are not predicted accurately. In one of the images, 0 is predicted as 9, but all other are correctly predicted.

show_img(new_images[:9,:], predicted_num)