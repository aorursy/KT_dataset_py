import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.listdir("../input"))

training_data = pd.read_csv('../input/train.csv')

testing_data = pd.read_csv('../input/test.csv')
# Transform DataFrame into array

training_data_X = training_data.iloc[:, 1:].values

training_data_y = training_data.iloc[:, 0].values

testing_data = testing_data.iloc[:, :].values
# Data Normalization

import tensorflow as tf

import tensorflow.keras as keras

from tensorflow.keras.layers import Dense, Flatten, Activation, MaxPooling2D, Conv2D
training_data_X = tf.keras.utils.normalize(training_data_X, axis=1)

testing_data = tf.keras.utils.normalize(testing_data, axis=1)
# Transform 1*784array into 28*28array
print(training_data_X.shape)

print(testing_data.shape)
training_data_X = training_data_X.reshape(42000, 28, 28)

testing_data = testing_data.reshape(28000, 28, 28)
print(training_data_X.shape)

print(testing_data.shape)
# Data Visualization 

import matplotlib.pyplot as plt
plt.imshow(training_data_X[0], cmap='gray')
# Creat CNN Model
# Keras CNN requires an extra dimension in the end which correspond to channels. 

# MNIST images are gray scaled so it use only one channel.For RGB images, there is 3 channels



training_data_X = training_data_X.reshape(-1, 28, 28, 1)

testing_data = testing_data.reshape(-1, 28, 28, 1)
print(training_data_X.shape)

print(testing_data.shape)
model = tf.keras.Sequential()
model.add(Conv2D(50, (3, 3), input_shape=(28, 28, 1), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(50, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(50, (3, 3)))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(256, activation=tf.nn.relu))

model.add(Dense(10, activation=tf.nn.softmax))
# Compile the model
model.compile(optimizer='Adam',

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])
model.fit(training_data_X, training_data_y, batch_size=500, epochs=30, validation_split=0.3)
predictions = model.predict_classes(testing_data, verbose=0)



submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),

                         "Label": predictions})

submissions.to_csv("DR.csv", index=False, header=True)