import numpy as np

import os

import pickle

import sys

import tarfile



from keras import backend as K

import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation, Flatten

from keras.layers import Conv2D, MaxPooling2D
#Utility functions for loading the data from Kaggle

def untar(filename, directory):

    tar_file = tarfile.open(filename)

    tar_file.extractall(path=directory)



def load_batch(directory):

    with open(directory, 'rb') as f:

        if sys.version_info < (3,):

            d = pickle.load(f)

        else:

            d = pickle.load(f, encoding='bytes')

            # decode utf8

            d_decoded = {}

            for k, v in d.items():

                d_decoded[k.decode('utf8')] = v

            d = d_decoded

    data = d['data']

    labels = d['labels']

    data = data.reshape(data.shape[0], 3, 32, 32)

    return data, labels



def load_data(directory):

    if not os.path.exists(directory):

        untar('../input/cifar10-python/cifar-10-python.tar.gz','/tmp')

        

    y_train = []

    

    for i in range(1, 6):

        x_train_temp, y_train_temp = load_batch(directory + "/data_batch_" + str(i))

        y_train += y_train_temp

        

        if i==1:

            x_train = x_train_temp

        else:

            x_train = np.row_stack([x_train,x_train_temp])  

            

    x_test, y_test = load_batch(directory + "/test_batch")

    y_train = np.array(y_train)

    y_test = np.array(y_test)

    

    if K.image_data_format() == 'channels_last':    

        x_test = x_test.transpose(0, 2, 3, 1)

        x_train = x_train.transpose(0, 2, 3, 1)

    return x_train,y_train,x_test,y_test
directory ='/tmp/cifar-10-batches-py'



num_classes = 10

batch_size = 32

epochs = 50



x_train,y_train,x_test,y_test=load_data(directory)
# Normalizing data

x_train = x_train.astype('float32') / 255

x_test = x_test.astype('float32') / 255



# Applying one hot encode outputs

y_train = keras.utils.to_categorical(y_train, num_classes)

y_test = keras.utils.to_categorical(y_test, num_classes)
model = Sequential()



model.add(Conv2D(32, (3, 3), padding='same',input_shape=x_train.shape[1:]))

model.add(Activation('relu'))

model.add(Conv2D(32, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Conv2D(64, (3, 3), padding='same'))

model.add(Activation('relu'))

model.add(Conv2D(64, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(512))

model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(num_classes))

model.add(Activation('softmax'))



# initiate RMSprop optimizer and configure some parameters

opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)



# Let's create our model

model.compile(loss = 'categorical_crossentropy',

              optimizer = opt,

              metrics = ['accuracy'])



print(model.summary())
history = model.fit(x_train, 

                    y_train,

                    batch_size=batch_size,

                    epochs=epochs,

                    validation_data=(x_test, y_test),

                    shuffle=True)

scores = model.evaluate(x_test, y_test, verbose=1)



print('Test loss:', scores[0])

print('Test accuracy:', scores[1])
import matplotlib.pyplot as plt



history_dict = history.history



loss_values = history_dict['loss']

val_loss_values = history_dict['val_loss']

epochs = range(1, len(loss_values) + 1)



line1 = plt.plot(epochs, val_loss_values, label='Validation/Test Loss')

line2 = plt.plot(epochs, loss_values, label='Training Loss')

plt.setp(line1, linewidth=2.0, marker = '+', markersize=10.0)

plt.setp(line2, linewidth=2.0, marker = '4', markersize=10.0)

plt.xlabel('Epochs') 

plt.ylabel('Loss')

plt.grid(True)

plt.legend()

plt.show()
# Plotting our accuracy charts

import matplotlib.pyplot as plt



history_dict = history.history



acc_values = history_dict['acc']

val_acc_values = history_dict['val_acc']

epochs = range(1, len(loss_values) + 1)



line1 = plt.plot(epochs, val_acc_values, label='Validation/Test Accuracy')

line2 = plt.plot(epochs, acc_values, label='Training Accuracy')

plt.setp(line1, linewidth=2.0, marker = '+', markersize=10.0)

plt.setp(line2, linewidth=2.0, marker = '4', markersize=10.0)

plt.xlabel('Epochs') 

plt.ylabel('Accuracy')

plt.grid(True)

plt.legend()

plt.show()