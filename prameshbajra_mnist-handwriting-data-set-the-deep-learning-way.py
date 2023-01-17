import tensorflow as tf

import keras

import matplotlib.pyplot as plt

import numpy as np



from tensorflow.nn import relu,softmax



from keras.utils import normalize

from keras.models import Sequential

from keras.layers import Flatten, Dense
mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()



plt.imshow(x_train[10], cmap=plt.cm.binary)

plt.show()



x_train = normalize(x_train, axis = 1)

x_test = normalize(x_test, axis = 1)



plt.imshow(x_train[10], cmap=plt.cm.binary)

plt.show()
# This will be the first layer of neural network ... Input layer ...

model = Sequential()

# Need to convert to 1D array before sending to dense layers ...

model.add(Flatten())

# Fully Connected layers up ahead ...

model.add(Dense(128, activation=relu))

# Output layer ... 10 outputs might be there ... 

model.add(Dense(10, activation=softmax))
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
model.fit(x_train, y_train, epochs = 3)
loss, accuracy = model.evaluate(x_test, y_test)

print(loss, accuracy)
plt.imshow(x_test[14])

plt.show()



predictions = model.predict(x_test)

print("Predicted :::> {}".format(np.argmax(predictions[14])))