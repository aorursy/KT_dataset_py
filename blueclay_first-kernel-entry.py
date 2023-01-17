import os, shutil

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

%matplotlib inline

from keras import layers

from keras import models

from keras.utils import to_categorical

from keras import optimizers
# Use pandas Dataframe to read from csv file

base_dir = '.'

#train_dir = os.path.join(base_dir, 'train.csv')

train_df = pd.read_csv("../input/train.csv")

#test_dir = os.path.join(base_dir, 'test.csv')

test_df = pd.read_csv('../input/test.csv')



# Generate training, tests and label ndarrays from dataframes

train_images = train_df.drop(labels=['label'], axis=1).values

test_images = test_df.values

train_labels = train_df['label'].values



print('train_images shape: ', train_images.shape)

print('train_labels shape: ', train_labels.shape)

print('test_images shape: ', test_images.shape)



# Now reshape the data to a shape that the model expects.

# Convert flattened training images (42000x784) to 42000x28x28x1 

train_images = np.resize(train_images, (train_images.shape[0], 28, 28, 1))

print('train_images shape: ', train_images.shape)



# Convert flattened test images (42000x784) to 42000x28x28x1

test_images = np.resize(test_images, (test_images.shape[0], 28, 28, 1))

print('test_images shape: ', test_images.shape)
# Transform 0 to 255 interval to values between 0 and 1 

train_images = train_images.astype('float32')/255

test_images = test_images.astype('float32')/255



# Use the keras one-hot encoding on the labels

train_labels = to_categorical(train_labels)

print ('training labels shape: ',train_labels.shape)
# Create our model

model = models.Sequential()



# Add Convolution layers

model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)))

model.add(layers.MaxPooling2D(2,2))

model.add(layers.Conv2D(64, (3,3), activation='relu'))

model.add(layers.MaxPooling2D(2,2))



# Add fully connected layers

model.add(layers.Flatten())

model.add(layers.Dense(64, activation='relu'))



# Ending the network with Dense layer of size 10. 

# This means for each input sample, our network will output

# a 10-dimentional vector representing one output class of 1 to 9.

# Th last layer uses the Softmax activation will output a 

# probability distribution over 10 different output classes. 

# The 10 scores will sum to 1.

model.add(layers.Dense(10, activation='softmax'))



model.summary()
# Default RMSProp parameters

optimizers = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

model.compile(optimizer=optimizers,

             loss='categorical_crossentropy',

             metrics=['accuracy'])
# Trains the model for a fixed number of epochs.

model.fit(train_images, train_labels, epochs=5, batch_size=64, verbose=0)
# Predict our test labels 

test_labels = model.predict(test_images)
# Create the submission array by getting the column index vector 

# to 1D with the highest probability value 

submission = np.argmax(test_labels, axis=1)
# Create a submission csv file

df = pd.DataFrame(data =(np.arange(1,submission.shape[0]+1)))

df.columns = ['ImageId']

df = df.assign(Label=submission)

df.to_csv('submission.csv', index=False)