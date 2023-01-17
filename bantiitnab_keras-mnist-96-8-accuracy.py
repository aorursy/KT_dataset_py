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
# importing some libraries

import keras

from keras.layers import Dense, Dropout

from keras.models import Sequential

from keras.optimizers import RMSprop

# load the data

train_data = pd.read_csv("../input/train.csv")

test_data = pd.read_csv("../input/test.csv")
# define some variables

num_classes = 10



# get x_train and y_train

x_train = train_data.iloc[:, 1:].values.astype('float32')

x_train /= 256

y_train = train_data.iloc[:, 0:1].values.astype('float32')



# get x_test

x_test = test_data.values.astype('float32') / 256



# convert y_train to 1 hot encoding

y_train = keras.utils.to_categorical(y_train, num_classes)



# print them

# print(x_train[0, :])

# print(y_train[0, :])

print(x_train.shape)

print(y_train.shape)

print(test_data.shape)

# build the model architecture

model = Sequential()

model.add(Dense(512, activation='relu', input_shape=(784,)))

model.add(Dropout(0.2))

model.add(Dense(512, activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(num_classes, activation='softmax'))

model.summary()
# train model

epochs = 2

batch_size = 64

model.compile(loss="categorical_crossentropy",

             optimizer=RMSprop(),

             metrics=['accuracy'])



history = model.fit(x_train, y_train, 

                   batch_size=batch_size,

                   epochs=epochs,

                   verbose=1,

                   validation_split=0.2)

# do prediction

predictions = model.predict(x_test, batch_size=64, verbose=1)

# print(predictions)

# with open("check.txt", "w") as fp:

#     fp.write("hello")
# print(check_output(["ls", "."]).decode("utf8"))

# saving results into file

print(prediction.shape)

with open("output.csv", "w") as fp:

    fp.write("ImageId,Label")

    for i,prediction in enumerate(predictions):

        pred_class = prediction.argmax()

        fp.write("{},{}\n".format(i, pred_class))

        