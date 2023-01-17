# Set seed with numpy
import numpy as np
np.random.seed(42)
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
# Kaggle does not allow us to download the dataset through Keras
# Luckily it is availeble on Kaggle anyways
# This notebook is connected to the dataset so we can load it like this
def mnist_load_data(path='mnist.npz'):
    with np.load(path) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']
    return (x_train, y_train), (x_test, y_test)
        

(X_train, y_train), (X_test, y_test) = mnist_load_data(path='../input/mnist.npz')
# Visualize MNIST
pixels = X_train[0]
label = y_train[0]
# Reshape the array into 28 x 28 array (2-dimensional array)
pixels = pixels.reshape((28, 28))

# Plot
plt.title('Label is {label}'.format(label=label))
plt.imshow(pixels, cmap='gray')
plt.show()
from sklearn.preprocessing import OneHotEncoder
# Generate one hot encoding

# Reshape from array to vector
y_train = y_train.reshape(y_train.shape[0],1)
# Generate one hot encoding
enc = OneHotEncoder()
onehot = enc.fit_transform(y_train)
# Convert to numpy vector
y_train = onehot.toarray()

# Reshape from array to vector
y_test = y_test.reshape(y_test.shape[0],1)
# Generate one hot encoding
enc = OneHotEncoder()
onehot = enc.fit_transform(y_test)
# Convert to numpy vector
y_test = onehot.toarray()
# Visualize MNIST
# NOW WITH ONE HOT
pixels = X_train[0]
label = y_train[0]
# Reshape the array into 28 x 28 array (2-dimensional array)
pixels = pixels.reshape((28, 28))

# Plot
plt.title('Label is {label}'.format(label=label))
plt.imshow(pixels, cmap='gray')
plt.show()
X_train = X_train.reshape(X_train.shape[0],X_train.shape[1] * X_train.shape[2])
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] * X_test.shape[2])
model = Sequential()
# For the first layer we have to specify the input dimensions
model.add(Dense(units=320, input_dim=784, activation='tanh'))

model.add(Dense(units=160, activation='tanh'))

model.add(Dense(units=10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
# x_train and y_train are Numpy arrays --just like in the Scikit-Learn API.
history = model.fit(X_train, y_train, epochs=10, batch_size=32)
# Plot the loss development
plt.plot(history.history['loss'])
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()
# Plot the accuracy development
plt.plot(history.history['acc'])
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()
model.evaluate(x=X_test,y=y_test)
from keras import optimizers
# Same Sequential model
model = Sequential()
# Add layers
model.add(Dense(units=320, input_dim=784, activation='tanh'))
model.add(Dense(units=160, activation='tanh'))
model.add(Dense(units=10, activation='softmax'))
# New compile statement
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=0.001),
              metrics=['accuracy'])
# Training should be much more slow now
# x_train and y_train are Numpy arrays --just like in the Scikit-Learn API.
history = model.fit(X_train, y_train, epochs=10, batch_size=32)
plt.plot(history.history['acc'])
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()
model.evaluate(x=X_test,y=y_test)
# On Kaggle this can be found under Outputs
model.save('my_model.h5')
# First we need to import the corresponding function
from keras.models import load_model
model = load_model('my_model.h5')