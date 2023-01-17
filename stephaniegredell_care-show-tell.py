# Import dependencies
import numpy as np # linear algebra
import os
import pandas as pd

# Keras is the primary library we will use. It is a part of Tensorflow but can also be used standalone.
import keras.utils
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from matplotlib import pyplot as plt

print("dependencies imported")
# Function to load data from our input path
def load_data(path):
    with np.load(path) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']
        return (x_train, y_train), (x_test, y_test)


# X_train is the data our model will learn
# y_train is the label
# X_valid and y_valid are the validation sets. We use these sets to validate the accuracy of our trained model.
# The capitalized first letter of the variable denotes that it's a two dimensional or higher data structure. 
# The first letter of the variable being lowercase would indicate it's one dimen
(X_train, y_train), (X_valid, y_valid) = load_data('../input/keras-mnist-amazonaws-npz-datasets/mnist.npz')
print("data loaded.")
# This will show us how many images we have in our dataset
X_train.shape

# 60000 images which are each 28x28 matrix of value
# This will show us that we have 60000 labels that corresponds to the training data 
y_train.shape
# This will show you the first 12 labels
y_train[0:12]
# Define the size of your figure
plt.figure(figsize=(3,3))

# Loop through the first dozen
for k in range(12):
    
    # Add subplot to current figure
    plt.subplot(3, 4, k+1)
    
    # Display data as image
    plt.imshow(X_train[k], cmap='Greys')
    
    # Turn off axis lines
    plt.axis('off')
    
# Adjust plot layers to give padding
plt.tight_layout()

# Show figure
plt.show()
# This tells us the amount and size of our validation dataset
X_valid.shape
# This tells us the number of labels in our validation set
y_valid.shape
# If you want to see how these relate, see the image this puts out
plt.imshow(X_valid[0], cmap='Greys')
# And then see the number this puts out
y_valid[0]
# Flatten 2-dimensional to 1-dimensional as well as convert it to float values
X_train = X_train.reshape(60000, 784).astype('float32')
X_valid = X_valid.reshape(10000, 784).astype('float32')

X_train /= 255
X_valid /= 255

X_valid[0]
# Convert to one hot representation
n_classes = 10
y_train = keras.utils.to_categorical(y_train, n_classes)
y_valid = keras.utils.to_categorical(y_valid, n_classes)

y_valid[0]
# Create a model. This is the simplest of models. 
# It is called sequential because each layer of neurons only passes data to the next layer in a sequential manner.
model = Sequential()

# Specify attributes of our hidden layer
model.add(Dense(64, activation='sigmoid', input_shape=(784,)))

# Specify attributes of the output layer
model.add(Dense(10, activation='softmax'))

# This will give us a summary of our model's schema.
model.summary()
# Compile the model
model.compile(loss='mean_squared_error', optimizer=SGD(lr=0.01), metrics=['accuracy'])
print('model compiled')
# Use the fit method to train the model
model.fit(X_train, y_train, batch_size=128, epochs=200, verbose=1, validation_data=(X_valid, y_valid))
# To validate the accuracy of our model, we pass in our validation data set
model.evaluate(X_valid, y_valid)