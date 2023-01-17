# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import os

import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline



# to get the accuracy score, provided actual class labels and predicted labels

from sklearn.metrics import accuracy_score



# to split the dataset into training and validation set

from sklearn.model_selection import train_test_split



# for reproducibility purposes

seed = 42

np.random.RandomState(seed)



print(os.listdir("../input/digit-recognizer"))
# importing the various Kers modules



# We will be using the SGD optimization technique.

# This method calculate the cost function for each datapoint individually.

# Unlike th normal GD, in which error is calculated on the entire dataset altogether.

from keras.optimizers import SGD 



# This is used to create a linear stack of layers, starting from input layer, 

# followed by hidden layers  and ending with the output layer.

from keras.models import Sequential, Model



# to convert the categorical data into One-hot encoding vector

from keras.utils import to_categorical



# Dense is to create a single layer in a network, consist of input shape, number of nodes in the layer

# and the activation function to use for the nodes in the layer

# Dropout is to specify the dropout rate after each layer

from keras.layers import Dense, Dropout, Flatten, Input, Bidirectional, LSTM, GlobalMaxPool1D, Lambda, Concatenate



# we will require permute_dimensions to transpose the input/image

from keras.backend import permute_dimensions



# History is used to get the various information about the training of the model, 

# like the training and test accuracies at the each epoch



# EarlyStopping help us to stop the training is the model is not showing significant improvement,

# in certain number of epochs

from keras.callbacks import EarlyStopping, History
# some of the constants

IMG_SIZE = 28 # size of the image

NUM_CLASSES = 10 # number of digits/classes
# to read the training data i.e. the MNIST image data

train_df = pd.read_csv("../input/digit-recognizer/train.csv")

train_df.head(5)
print("Shape of the training dataset: ", train_df.shape)
# seperating the labels from the image pixels information

y = train_df.values[:, 0] # get the values in the first column

X = train_df.values[:, 1:]/255.0 # get all the columns but the first. Also, we divided by 255 to normalize the values



# reshape the X into 28x28 size

X = X.reshape(-1, IMG_SIZE, IMG_SIZE)
# plotting a sample non transposed image

plt.imshow(X[2])
# we will create another set of images, in which the images will be transposed

# the reason is that we are going to train 2 LSTM models,

# one will read the image from left to right, treating each row in a image as a single sequence

# also, we will read the image from top to bottom



# axes array's size is equal to the number of dimensions in an image

# our X array is 3 dimensional, first is the number of images and other two the 2-dimensional image

# (0, 2, 1) - tells to keep the first dimension as 1st dim and interchange the other two

X_T = np.transpose(X, axes=(0, 2, 1))



# plotting a sample transposed image

plt.imshow(X_T[2])
print("Size of the un-transposed dataset: ", X.shape[0])

print("Size of the transposed dataset: ", X_T.shape[0])
# create dummy variables from the labels using One-hot encoding method, which we imported above

y_encoded = to_categorical(y)

print("Shape of the target variable: ", y_encoded.shape)



print("First few records:")

print(y_encoded[:2])
# Let's first declare some constants



# split the data into train and validation datasets with test size to be 20% of the total

VALIDATION_SIZE = .2

# number of times model is trained on the entire dataset

EPOCHS = 10

# number of images to train at a single time

BATCH_SIZE = 32

# size of the LSTM hidden/cell state

LATENT_DIM = 50
# the input to the Bi-LSTM will be an image at a time

image_input = Input(shape=(IMG_SIZE, IMG_SIZE))



# here we are implementing the bidirectional LSTM, which will read the images both left to right and right to left

# where, units - is the dimension of the hidden/cell state

# and return_sequences - returns the output (hidden state) for each cell unit



# so, size of each input to a cell is - 1x28 and there are 28 such sequences

# and output of each cell is 1xLATENT_DIMx2 (since bidirectional)

l2r_lstm = Bidirectional(LSTM(units=LATENT_DIM, return_sequences=True))(image_input)



# the output size from previous layer is - 28xLATENT_DIMx2

# but this contains a lot of repetitive information

# so, we will take the maximum value for each of the LATENT_DIM

# hence the output size will be - LATENT_DIMx2

l2r_lstm = GlobalMaxPool1D()(l2r_lstm)



# this completes reading image from Left to Right
# to read the input from top to bottom, we will transpose the input and then feed it to the Bi-LSTM model

# just like python-lambda, this reads an image and transpose it

# the pattern works just like what we did above to transpose the image

transpose_input = Lambda(lambda img: permute_dimensions(img, pattern=(0, 2, 1)))(image_input)



# add the bi-lstm layer

t2b_lstm = Bidirectional(LSTM(units=LATENT_DIM, return_sequences=True))(transpose_input)



# add the global max-pooling layer

t2b_lstm = GlobalMaxPool1D()(t2b_lstm)



# this completes reading image from Top to Bottom
# now, we will combine both the models and train it



# l2r_lstm output size - NxLATENT_DIMx2

# t2b_lstm output size - NxLATENT_DIMx2

# after concat the output size will be - NxLATENT_DIMx4

l2r_t2b_lstm = Concatenate(axis=1)([l2r_lstm, t2b_lstm])



# add the Dense layer with softmax function

# this will output the probabilities of a image being classified into one of the classes

output_probs = Dense(NUM_CLASSES, activation='softmax')(l2r_t2b_lstm)



bi_lstm_model = Model(inputs=image_input, outputs=output_probs)



# compile the model

bi_lstm_model.compile(

    optimizer="adam",

    loss="categorical_crossentropy",

    metrics=["accuracy"]

)
# print the model summary

bi_lstm_model.summary()
# train the model

history = bi_lstm_model.fit(

    X, y_encoded,

    validation_split=VALIDATION_SIZE, 

    epochs=EPOCHS, 

    batch_size=BATCH_SIZE

)
# plot the losses for both training and testing

plt.plot(history.history['loss'], label='loss')

plt.plot(history.history['val_loss'], label='val_loss')

plt.legend()

plt.show()



# plot the accuracies for both training and testing

plt.plot(history.history['accuracy'], label='acc')

plt.plot(history.history['val_accuracy'], label='val_acc')

plt.legend()

plt.show()
# read the test dataset and reshape it

X_predict = pd.read_csv("../input/digit-recognizer/test.csv")

X_predict = X_predict.values.reshape(-1, IMG_SIZE, IMG_SIZE) / 255.0 # normalize the values



# plotting an image

plt.imshow(X_predict[0])
# get the predicted labels

y_predicted_classes = np.argmax(bi_lstm_model.predict(X_predict), axis=1)



# print the predicted label of the above image

y_predicted_classes[0]
# create the submissions dataset and save it

submissions = pd.DataFrame()

submissions["ImageId"] = [i for i in range(1, y_predicted_classes.shape[0]+1)]

submissions["Label"] = y_predicted_classes



submissions.to_csv("submissions.csv", index=False)