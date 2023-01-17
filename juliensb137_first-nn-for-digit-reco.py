#import packages and functions

import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

from keras.utils import np_utils

from keras.models import Sequential

from keras.layers import Dense, Dropout, Lambda, Flatten

from keras.layers.convolutional import *

from keras.optimizers import Adam ,RMSprop

from sklearn.model_selection import train_test_split
#import data

train_file = pd.read_csv("../input/train.csv")

print (train_file.shape)

test_images = pd.read_csv("../input/test.csv")

print (test_images.shape)
#for CNN

x_train = train_file.drop(['label'], axis=1).values.astype('float32')

Y_train = train_file['label'].values

x_valid = test_images.values.astype('float32')
#for FNN

#remove labels and save it in a specific vector

#Label is the first column, rest is 28*28=784 pixel-columns

train_images = (train_file.ix[:,1:].values).astype('float32')

print (train_images.shape)

train_labels = train_file.ix[:,0].values.astype('int32')

print (train_labels.shape)
#for CNN

#reshape train and test

img_width, img_height = 28, 28

n_train = x_train.shape[0]

n_valid = x_valid.shape[0]

n_classes = 10 

x_train = x_train.reshape(n_train,1,img_width,img_height)

x_valid = x_valid.reshape(n_valid,1,img_width,img_height)



x_train = x_train/255 #normalize from [0,255] to [0,1]

x_valid = x_valid/255 

y_train = np_utils.to_categorical(Y_train)
#for FNN

#Convert train dataset to (num_images, img_rows, img_cols) format 

train_images = train_images.reshape((42000, 28 * 28))
#we need to normelize the pixel value :

train_images = train_images / 255

test_images = test_images / 255

#and transform label into categorie

train_labels = np_utils.to_categorical(train_labels)

num_classes = train_labels.shape[1]
#FNN !!!

# fix random seed for reproducibility

seed = 43

np.random.seed(seed)

#designing the network :

model=Sequential() #configures the learning process for a sequential model

#relu as activation function, first layer is input layer

model.add(Dense(64, activation='relu',input_dim=(28 * 28)))

#model.add(Dense(32,activation='relu',input_dim=(28 * 28)))

#model.add(Dense(16,activation='relu'))

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.15))

model.add(Dense(64, activation='relu'))

model.add(Dropout(0.15))

model.add(Dense(32, activation='relu'))

model.add(Dropout(0.15))

#output layer, a 10 classes problem so output = 10!

model.add(Dense(10,activation='softmax'))
import pandas as pd

import numpy as np

from keras.utils.np_utils import to_categorical

from keras import backend as K



K.set_image_dim_ordering('th') #input shape: (channels, height, width)



train_df = pd.read_csv("../input/train.csv")

valid_df = pd.read_csv("../input/test.csv")



x_train = train_df.drop(['label'], axis=1).values.astype('float32')

Y_train = train_df['label'].values

x_valid = valid_df.values.astype('float32')



img_width, img_height = 28, 28



n_train = x_train.shape[0]

n_valid = x_valid.shape[0]



n_classes = 10 



x_train = x_train.reshape(n_train,1,img_width,img_height)

x_valid = x_valid.reshape(n_valid,1,img_width,img_height)



x_train = x_train/255 #normalize from [0,255] to [0,1]

x_valid = x_valid/255 



y_train = to_categorical(Y_train)





from keras.models import Sequential

from keras.layers.convolutional import *

from keras.layers.core import Dropout, Dense, Flatten, Activation



n_filters = 64

filter_size1 = 3

filter_size2 = 2

pool_size1 = 3

pool_size2 = 1

n_dense = 128



model = Sequential()



model.add(Convolution2D(n_filters, filter_size1, filter_size1, batch_input_shape=(None, 1, img_width, img_height), activation='relu', border_mode='valid'))



model.add(MaxPooling2D(pool_size=(pool_size1, pool_size1)))



model.add(Convolution2D(n_filters, filter_size2, filter_size2, activation='relu', border_mode='valid'))



model.add(MaxPooling2D(pool_size=(pool_size2, pool_size2)))



model.add(Dropout(0.25))



model.add(Flatten())



model.add(Dense(n_dense))



model.add(Activation('relu'))



model.add(Dropout(0.5))



model.add(Dense(n_classes))



model.add(Activation('softmax'))



model.compile(optimizer='adam',

              loss='categorical_crossentropy',

              metrics=['accuracy'])

"""

#CNN !!

n_filters = 64

filter_size1 = 3

filter_size2 = 2

pool_size1 = 3

pool_size2 = 1

n_dense = 128

n_classes =10

img_width, img_height = 28, 28



model = Sequential()

model.add(Convolution2D(n_filters, filter_size1, filter_size1, batch_input_shape=(None, 1, img_width, img_height), activation='relu', border_mode='valid'))

model.add(MaxPooling2D(pool_size=(pool_size1, pool_size1)))

model.add(Convolution2D(n_filters, filter_size2, filter_size2, activation='relu', border_mode='valid'))

model.add(MaxPooling2D(pool_size=(pool_size2, pool_size2)))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(64))

model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(n_classes))

model.add(Dense(n_classes,activation='softmax'))

#model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

"""
#defining metric, loss function and optimize

model.compile(optimizer=RMSprop(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
batch_size = 128

n_epochs = 2



model.fit(x_train,

          y_train,

          batch_size=batch_size,

          epochs=n_epochs,verbose=2,

          validation_split=.2)
#fitting the model

history=model.fit(train_images, train_labels, validation_split = 0.05, nb_epoch=25, batch_size=64)
#print the network definition

print(model.summary())
test_images2 = (test_images.values).astype('float32')

test_images2 = test_images2.reshape((28000, 28 * 28))

predictions = model.predict(test_images2)
#CNN

predictions = model.predict(x_valid)
predictions = np_utils.categorical_probas_to_classes(predictions)
np.savetxt('mnist_output.csv', np.c_[range(1,len(predictions)+1),predictions], delimiter=',', header = 'ImageId,Label', comments = '', fmt='%d')
output_file = "submission.csv"

out = np.column_stack((range(1, predictions.shape[0]+1), predictions))

np.savetxt(output_file, out, header="ImageId,Label", comments="", fmt="%d,%d")
history_dict = history.history

history_dict.keys()
import matplotlib.pyplot as plt

%matplotlib inline

loss_values = history_dict['loss']

val_loss_values = history_dict['val_loss']

epochs = range(1, len(loss_values) + 1)



# "bo" is for "blue dot"

plt.plot(epochs, loss_values, 'bo')

# b+ is for "blue crosses"

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