%matplotlib inline

import numpy as np

import matplotlib.pyplot as plt



from keras import regularizers

from keras.datasets import mnist

from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation, Flatten

from keras.layers import Conv2D, MaxPooling2D

from keras.utils import np_utils

from keras import optimizers
# import mnist data and visualize first image

(X_train, y_train), (X_test, y_test) = mnist.load_data()

plt.imshow(X_train[0])

print(y_train[0])
# scale data and reshape

X_train = X_train.astype('float32')

X_test = X_test.astype('float32')



X_train /= 255

X_test /= 255



X_train = X_train.reshape(-1,28,28,1)

X_test = X_test.reshape(-1,28,28,1)



# make output a 10 dim vector indicating class

Y_train = np_utils.to_categorical(y_train, 10)

Y_test = np_utils.to_categorical(y_test, 10)
# create convolution neural network

model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3),

                 activation='relu',

                 input_shape=(28,28,1)))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(.50))

model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(.25))

model.add(Flatten())

model.add(Dropout(.5))

model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(.0001)))

model.add(Dropout(.25))

model.add(Dense(10, activation='softmax'))
model.summary()
# compile model

model.compile(loss='categorical_crossentropy', 

              optimizer='adam',

              metrics=['accuracy'])
hist = model.fit(X_train, Y_train,

          batch_size=500, nb_epoch=10,verbose=1,

          validation_data=(X_test, Y_test))
# make predictions

Y_test_pred = model.predict_classes(X_test)
plt.plot(hist.history['val_acc'])

plt.title('Model Validation Accuracy')

plt.ylabel('Validation Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()