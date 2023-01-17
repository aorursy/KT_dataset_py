from __future__ import absolute_import

from __future__ import division

from __future__ import print_function

from __future__ import unicode_literals



import tensorflow as tf



tf.enable_eager_execution()



import tensorflow.keras

from tensorflow.keras.datasets import mnist

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout, Flatten

from tensorflow.keras.layers import Conv2D, MaxPooling2D

from tensorflow.keras import backend as K



import warnings

warnings.filterwarnings('ignore')



import matplotlib.pyplot as plt

%matplotlib inline



batch_size = 128

num_classes = 10

epochs = 14
# input image dimensions

img_rows, img_cols = 28, 28



# the data, split between train and test sets

(x_train, y_train), (x_test, y_test) = mnist.load_data()



if K.image_data_format() == 'channels_first':

    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)

    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)

    input_shape = (1, img_rows, img_cols)

else:

    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)

    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

    input_shape = (img_rows, img_cols, 1)



x_train = x_train.astype('float32')

x_test = x_test.astype('float32')

x_train /= 255

x_test /= 255

print('x_train shape:', x_train.shape)

print(x_train.shape[0], 'train samples')

print(x_test.shape[0], 'test samples')



# convert class vectors to binary class matrices

y_train = tensorflow.keras.utils.to_categorical(y_train, num_classes)

y_test = tensorflow.keras.utils.to_categorical(y_test, num_classes)
def plot_history(history):

    loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]

    val_loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' in s]

    acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' not in s]

    val_acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' in s]

    

    if len(loss_list) == 0:

        print('Loss is missing in history')

        return 

    

    epochs = range(1,len(history.history[loss_list[0]]) + 1)

    

    ## Loss

    plt.figure(1)

    for l in loss_list:

        plt.plot(epochs, history.history[l], 'b', label='Training loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))

    for l in val_loss_list:

        plt.plot(epochs, history.history[l], 'g', label='Validation loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))

    

    plt.title('Loss')

    plt.xlabel('Epochs')

    plt.ylabel('Loss')

    plt.legend()

    

    ## Accuracy

    plt.figure(2)

    for l in acc_list:

        plt.plot(epochs, history.history[l], 'b', label='Training accuracy (' + str(format(history.history[l][-1],'.5f'))+')')

    for l in val_acc_list:    

        plt.plot(epochs, history.history[l], 'g', label='Validation accuracy (' + str(format(history.history[l][-1],'.5f'))+')')



    plt.title('Accuracy')

    plt.xlabel('Epochs')

    plt.ylabel('Accuracy')

    plt.legend()

    plt.show()
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3),

                 activation='relu',

                 input_shape=input_shape))

model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dense(num_classes, activation='softmax'))



model.compile(loss=tensorflow.keras.losses.categorical_crossentropy,

              optimizer=tensorflow.keras.optimizers.Adam(),

              metrics=['accuracy'])



history = model.fit(x_train, y_train,

          batch_size=batch_size,

          epochs=epochs,

          verbose=1,

          validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score[0])

print('Test accuracy:', score[1])
plot_history(history)
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3),

                 activation='relu',

                 input_shape=input_shape))

model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.25))

model.add(Dense(num_classes, activation='softmax'))



model.compile(loss=tensorflow.keras.losses.categorical_crossentropy,

              optimizer=tensorflow.keras.optimizers.Adam(),

              metrics=['accuracy'])



history = model.fit(x_train, y_train,

          batch_size=batch_size,

          epochs=epochs,

          verbose=1,

          validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score[0])

print('Test accuracy:', score[1])
plot_history(history)
from dropin_util import Dropin
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3),

                 activation='relu',

                 input_shape=input_shape))

model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropin(0.75))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropin(0.75))

model.add(Dense(num_classes, activation='softmax'))



model.compile(loss=tensorflow.keras.losses.categorical_crossentropy,

              optimizer=tensorflow.keras.optimizers.Adam(),

              metrics=['accuracy'])



history = model.fit(x_train, y_train,

          batch_size=batch_size,

          epochs=epochs,

          verbose=1,

          validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score[0])

print('Test accuracy:', score[1])
plot_history(history)