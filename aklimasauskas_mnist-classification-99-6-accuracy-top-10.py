import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from keras.utils.np_utils import to_categorical
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
# Read the data
train = pd.read_csv('../input/digit-recognizer/train.csv').to_numpy()
test = pd.read_csv('../input/digit-recognizer/test.csv').to_numpy()
# Separate the images from their labels
X = train[:, 1:]
y = train[:, 0]

# Normalize the image data
X = X / 255
test = test / 255
X = X.reshape(42000, 28, 28, 1) # Set the data to be the right shape for the NN
yh = to_categorical(y, 10) # One-hot encode the labels
# Add data augmentation
idg = ImageDataGenerator(rotation_range=15, width_shift_range=4, height_shift_range=4)
idg.fit(X)
ensemble = []
# Make 15 CNNs.
# The CNNs consist of 6 convolutional layers, 2 of which (with strides=2) perform downsampling. These are followed by a fully-connected layer and an output layer.
# They use ReLU as the activation function (except for the output layer). The CNNs also use batch normalization and dropout for regularization.
for i in range(15):    
    cnn = keras.Sequential()

    cnn.add(layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=(28, 28, 1)))
    cnn.add(layers.BatchNormalization())
    
    cnn.add(layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
    cnn.add(layers.BatchNormalization())
    
    cnn.add(layers.Conv2D(filters=32, kernel_size=5, strides=2, activation='relu'))
    cnn.add(layers.BatchNormalization())
    cnn.add(layers.Dropout(0.25))

    cnn.add(layers.Conv2D(filters=64, kernel_size=3, activation='relu'))
    cnn.add(layers.BatchNormalization())
    
    cnn.add(layers.Conv2D(filters=64, kernel_size=3, activation='relu'))
    cnn.add(layers.BatchNormalization())
    
    cnn.add(layers.Conv2D(filters=64, kernel_size=5, strides=2, activation='relu'))
    cnn.add(layers.BatchNormalization())
    cnn.add(layers.Dropout(0.25))

    cnn.add(layers.Flatten())
    cnn.add(layers.Dense(256, activation='relu'))
    cnn.add(layers.BatchNormalization())
    cnn.add(layers.Dropout(0.5))

    cnn.add(layers.Dense(10, activation='softmax'))
    
    ensemble.append(cnn)   
steps = np.ceil(len(X) / 32) # Calculate the number of steps in an epoch (using batches of size 32)

# Train the CNNs (this might take several hours)
for i in range(15):
    
    # Exponential decay schedule for the learning rate. Values for initial_learning_rate and decay_rate were found using grid search.
    opt = keras.optimizers.Adam(learning_rate=ExponentialDecay(initial_learning_rate=0.003, decay_steps=steps, decay_rate=0.9))

    ensemble[i].compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    ensemble[i].fit(idg.flow(X, yh), epochs=30, verbose=0)
    
    print(str(i+1) + 'th NN trained')
probs = np.zeros((28000, 10))
# Add up predictions (probabilities) from all the CNNs
for i in range(15):
    probs += ensemble[i].predict(test.reshape(28000, 28, 28, 1))
y_hat = np.argmax(probs, axis=1) # Make the predictions
# Generate the submission output
sub = pd.DataFrame(y_hat, index=list(range(1, 28001)), columns=['Label'])
sub.index.name = 'ImageId'
sub.to_csv('sub.csv')