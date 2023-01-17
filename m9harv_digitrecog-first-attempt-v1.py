# Put these at the top of every notebook, to get automatic reloading and inline plotting

%reload_ext autoreload

%autoreload 2

%matplotlib inline



# Importing required libraries

import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

import keras

import sklearn

from keras.models import Sequential

from keras.layers import Dense, Activation, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D

from keras.optimizers import SGD



from sklearn.model_selection import train_test_split
#load CSVs



PATH = "../input/"

train_csv = f'{PATH}train.csv'

test_csv = f'{PATH}test.csv'

trainset=pd.read_csv(train_csv)

x_test=pd.read_csv(test_csv)
# visualize data

trainset.head()
#find out the column names of the training dataframe so we can seperate the labels

#from the training examples

trainset.columns
# creates a dataframe with the training data outputs

y_train = trainset['label']

# make sure it is correct 

y_train.shape

#This line shows us the distribution of the data

dist = y_train.value_counts(sort=False)

# transpose the list

dist = dist.T

#print(dist)

type(dist)

distDF = pd.DataFrame([dist])

print(distDF)
# plotting the distribution

#sns.set_style("whitegrid")

ax = sns.barplot(data=distDF,)
# convert y_train to array for keras 

y_train = y_train.values

type(y_train)





# to build x_train we copy the original dataframe

trainset_copy = trainset

# delete the labels from the copy

del trainset_copy['label']

# rename the result to x_train

x_train = trainset_copy
# input image dimensions

img_rows, img_cols = 28, 28





x_train = x_train.values

x_test = x_test.values

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)

x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
# create cross validation set

x_train, x_crossval, y_train, y_crossval = train_test_split(x_train, y_train, test_size = 0.1)

# visualize data

x_test.shape

x_train.shape

# initialize some variables

batch_size = 128

num_classes = 10

epochs = 12

input_shape = (img_rows, img_cols, 1)

#normalize data

#convert to floating point

x_train = x_train.astype('float32')

x_crossval = x_crossval.astype('float32')

x_test = x_test.astype('float32')

#divide by 255 to get values between 0 and 1

x_train /= 255

x_crossval /= 255

x_test /= 255
# convert class vectors to binary class matrices

y_train = keras.utils.to_categorical(y_train, num_classes)

y_crossval = keras.utils.to_categorical(y_crossval, num_classes)
# creating the keras sequential model CNN

#following the documentation example online



model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3),

                 activation='relu',

                 input_shape=input_shape))

model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(num_classes, activation='softmax'))



model.compile(loss=keras.losses.categorical_crossentropy,

              optimizer=keras.optimizers.Adadelta(),

              metrics=['accuracy'])



model.fit(x_train, y_train,

          batch_size=batch_size,

          epochs=epochs,

          verbose=1,

          validation_data=(x_crossval, y_crossval))

score = model.evaluate(x_crossval, y_crossval, verbose=0)

print('Test loss:', score[0])

print('Test accuracy:', score[1])
# the predictions

pred = model.predict_classes(x_test, verbose=1)

print(pred)
pred
submission = pd.DataFrame(data= {'ImageId': list(range(1,28001)), 'Label': pred })

print(submission)
submission.to_csv("m9harv_digitrecog.csv", index=False, header=True)