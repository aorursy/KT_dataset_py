# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.


import matplotlib.pyplot as plt

%matplotlib inline

train = pd.read_csv('../input/train.csv')



train_images = (train.ix[:,1:].values).astype('float32')

train_labels = train.ix[:,0].values.astype('int32')



print(train_images.shape)

test_images = (pd.read_csv('../input/test.csv').values).astype('float32')

train_images = train_images.reshape(train_images.shape[0], 28, 28)



plt.subplot(333)

plt.imshow(train_images[1], cmap=plt.get_cmap('gray'))

plt.title(train_labels[1])

train_images = train_images.reshape(train_images.shape[0], 784)

print (train_images.shape)

print (test_images.shape)
train_labels.shape
train_labels
train_images = train_images / 255

test_images = test_images / 255

# one-hot

from keras.utils.np_utils import to_categorical

train_labels = to_categorical(train_labels)

train_labels.shape
num_classes = train_labels.shape[0]
plt.title(train_labels[0])

plt.plot(train_labels[0])

plt.xticks(range(10))
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)

test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

train_images.shape

# convolution neural network



from keras.models import Sequential

from keras.layers import Convolution2D, MaxPooling2D, Dense, Activation, Dropout, Flatten

from keras.layers.advanced_activations import LeakyReLU



model = Sequential()

model.add(Convolution2D(32, (3, 3), padding='valid', input_shape=(28, 28, 1)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2), padding='same'))

model.add(Dropout(1))



model.add(Convolution2D(64, (3, 3)))

model.add(Activation('relu'))

model.add(Dropout(1))

model.add(MaxPooling2D(pool_size=(2,2), padding='same'))



model.add(Flatten())

model.add(Dense(1024, kernel_initializer='normal'))

model.add(Activation('relu'))

model.add(Dropout(1))



model.add(Dense(10, kernel_initializer='normal'))

model.add(Activation('softmax'))

from keras.optimizers import SGD

model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.05, momentum=0.9, nesterov=True), metrics=['accuracy'])

history = model.fit(train_images, train_labels, verbose=0,\

                    validation_split = 0.05, epochs=3, batch_size=64)

history_dict = history.history

history_dict.keys()
loss_values = history_dict['loss']

val_loss_values = history_dict['val_loss']

epochs = range(1, len(loss_values)+1)



plt.plot(epochs, loss_values, 'bo')

plt.plot(epochs, val_loss_values, 'b+')

plt.xlabel('Epochs')

plt.ylabel('Loss')



plt.show()
plt.clf()   # clear figure

acc_values = history_dict['acc']

val_acc_values = history_dict['val_acc']



plt.plot(epochs, acc_values, 'bo')

plt.plot(epochs, val_acc_values, 'b+')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')



plt.show()
predictions = model.predict_classes(test_images, verbose=0)



submissions = pd.DataFrame({'ImageId':list(range(1, len(predictions)+1)), \

                           'Label': predictions})

submissions.to_csv('DR.csv', index=False, header=True)

print(acc_values)

print(val_acc_values)