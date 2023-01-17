import pandas as pd

import numpy as np

from keras.datasets import mnist

from keras.models import Sequential, Model

from keras.layers import Dense, Dropout, Activation, Flatten

from keras.layers import MaxPooling2D, Conv2D



from keras.utils import np_utils

import requests

requests.packages.urllib3.disable_warnings()

import ssl
# Load pre-shuffled MNIST data into train and test sets

(X_train, y_train), (X_test, y_test) = mnist.load_data()
y_train[0]
%matplotlib inline

from matplotlib import pyplot as plt

plt.imshow(X_train[0])


X_train = X_train.reshape(X_train.shape[0], 28, 28,1)

X_test = X_test.reshape(X_test.shape[0], 28, 28,1)

print(X_train.shape)

print(X_test.shape)
X_train = X_train.astype('float32')

X_test = X_test.astype('float32')

X_train /= 255

X_test /= 255
Y_train = np_utils.to_categorical(y_train, 10)

Y_test = np_utils.to_categorical(y_test, 10)



print(Y_train.shape)
Y_train[0]
model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28,28,1)))
model.add(Conv2D(32, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])
model.summary()
history = model.fit(X_train, Y_train, 

          batch_size=32,validation_split=0.25, epochs=20, verbose=1)
# Plot training & validation accuracy values

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
score = model.evaluate(X_test, Y_test, verbose=0)
score
preds = model.predict(x=X_test[:10])
np.argmax(preds, axis=1)
np.argmax(Y_test[:10], axis=1)
plt.imshow(X_test.reshape(10000,28,28)[1])
intermediate_layer_model = Model(inputs=model.input,

                                 outputs=model.layers[1].output)



intermediate_output = intermediate_layer_model.predict(X_test[1].reshape(1,28,28,1))
intermediate_output.shape
intermediate_output = intermediate_output.reshape(24,24,32)
for i in range(intermediate_output.shape[2]):

    plt.figure()

    plt.imshow(intermediate_output[:,:,i-1])



plt.show()