# load the train and test datasets using pandas

import pandas as pd



trainSet = pd.read_csv('../input/train.csv')

testSet = pd.read_csv('../input/test.csv')



# assign the label column to a separate variable and remove it from the dataframe

labels = trainSet.pop('label')



# normalise the grayscale value from 0-255 to 0.0-1.0

trainSet /= 255

testSet /= 255
# define the hyperparameters of the DNN which consists of an input layer,

# two hidden layers and an output layer

INPUT_DIM = len(trainSet.columns)

NLAYER1 = 512

NLAYER2 = 64

OUTPUT_DIM = len(labels.unique())

EPOCHS = 5

BATCH_SIZE = 32
# build the model using Tensorflow & Keras

import tensorflow.keras as keras

from tensorflow.keras.layers import Input, Dense



model = keras.models.Sequential([

    #Input(shape=(INPUT_DIM,)), # this line helps with code readability but it doesn't work with kaggle's current version of tensorflow...

    Dense(NLAYER1, activation='relu', input_dim=INPUT_DIM),

    Dense(NLAYER2, activation='relu'),

    Dense(OUTPUT_DIM, activation='softmax'),

])



model.compile(loss='categorical_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])
# convert the labels to one-hot labels and train the model

one_hot_labels = keras.utils.to_categorical(labels, num_classes=10)

model.fit(trainSet, one_hot_labels, epochs=EPOCHS, batch_size=BATCH_SIZE)



# let the model label the test dataset!

result = model.predict(testSet)



# reverse one-hot labels back to ints

import numpy as np

result = np.argmax(result, axis=1)



# write to csv in the format required for the competition

submission = pd.DataFrame(result)

submission.index += 1

submission.to_csv('submission.csv', index_label='ImageId', header=['Label'])