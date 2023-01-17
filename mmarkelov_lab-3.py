import keras

from keras.datasets import fashion_mnist

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D

from keras import backend as K

import matplotlib.pyplot as plt

import tensorflow as tf

import numpy as np
batch_size = 128

num_classes = 10

epochs = 10

img_rows, img_cols = 28, 28

input_shape = (img_rows, img_cols, 1)
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
print(train_images.shape)



train_images = train_images.reshape(train_images.shape[0], img_rows, img_cols, 1)

test_images = test_images.reshape(test_images.shape[0], img_rows, img_cols, 1)



train_labels = keras.utils.to_categorical(train_labels, num_classes)

test_labels = keras.utils.to_categorical(test_labels, num_classes)



train_images = train_images / 255.0

test_images = test_images / 255.0
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3),

                 activation = 'relu',

                 input_shape = input_shape))

model.add(Conv2D(64, (3, 3), activation = 'relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(128, activation = 'relu'))

model.add(Dense(num_classes, activation = 'softmax'))
model.compile(loss = keras.losses.categorical_crossentropy,

              optimizer = keras.optimizers.Adadelta(),

              metrics = ['accuracy'])
model.fit(train_images, train_labels,

          batch_size=batch_size,

          epochs=epochs,

          verbose=1)
score = model.evaluate(test_images, test_labels, verbose = 0)

print('Test loss:', score[0])

print('Test accuracy', score[1])
from keras.datasets import cifar10



(x_train, y_train), (x_test, y_test) = cifar10.load_data()



print(x_train.shape)

print(y_train.shape)
batch_size = 128

num_classes = 10

epochs = 20

img_rows, img_cols = 32, 32

input_shape = (img_rows, img_cols, 3)



y_train = keras.utils.to_categorical(y_train, num_classes)

y_test = keras.utils.to_categorical(y_test, num_classes)



x_train = x_train / 255.0

x_test = x_test / 255.0
for i in range(9):

    plt.subplot(330 + 1 + i)

    plt.imshow(x_train[i])

plt.show()
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3),

                 activation = 'relu',

                 input_shape = input_shape))



model.add(Conv2D(64, (3, 3), activation = 'relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Flatten())

model.add(Dense(128, activation = 'relu'))

model.add(Dense(num_classes, activation = 'softmax'))



model.compile(loss = keras.losses.categorical_crossentropy,

              optimizer = keras.optimizers.Adadelta(),

              metrics = ['accuracy'])



model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1)
score = model.evaluate(x_test, y_test, verbose = 0)

print('Test loss:', score[0])

print('Test accuracy', score[1])



# Test loss: 0.8880965149879455

# Test accuracy 0.7258999943733215
y_pred = model.predict_classes(x_test)

y_test_cm = [np.argmax(y_test[i]) for i in range(0, y_test.shape[0])]

conf_matrix = tf.math.confusion_matrix(labels = y_test_cm,

                                       predictions = y_pred).numpy()

print(conf_matrix)
conf_matrix_percent = np.around(conf_matrix.astype('float')/conf_matrix.sum(axis = 1)[:np.newaxis], decimals = 2)

print(conf_matrix_percent)

print('')

for i in range(0, num_classes):

    print('class: ', i, ':', conf_matrix_percent[i,i])
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3),

                 activation = 'relu',

                 input_shape = input_shape))



model.add(Conv2D(64, (3, 3), activation = 'relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.2))

          

model.add(Flatten())

model.add(Dense(128, activation = 'relu'))

model.add(Dropout(0.2))

model.add(Dense(num_classes, activation = 'softmax'))



model.compile(loss = keras.losses.categorical_crossentropy,

              optimizer = keras.optimizers.Adadelta(),

              metrics = ['accuracy'])



model.fit(x_train, y_train,

          batch_size=batch_size,

          epochs=epochs,

          verbose=1)
score = model.evaluate(x_test, y_test, verbose = 0)

print('Test loss:', score[0])

print('Test accuracy', score[1])



# Test loss: 0.06216323704600334

# Test accuracy 0.985480010509491
y_pred = model.predict_classes(x_test)

y_test_cm = [np.argmax(y_test[i]) for i in range(0, y_test.shape[0])]

conf_matrix = tf.math.confusion_matrix(labels = y_test_cm,

                                       predictions = y_pred).numpy()



conf_matrix_percent = np.around(conf_matrix.astype('float')/conf_matrix.sum(axis = 1)[:np.newaxis], decimals = 2)

print(conf_matrix_percent)

print('')

for i in range(0, num_classes):

    print('class: ', i, ':', conf_matrix_percent[i,i])