# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from keras.models import Sequential                                   # model

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D   # layers

from keras.optimizers import RMSprop                                  # optimizer

from keras.preprocessing.image import ImageDataGenerator              

from keras.callbacks import ReduceLROnPlateau                         # callback function



import matplotlib.pyplot as plt

%matplotlib inline



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))                #list the input folder files
# create the training & test sets, skipping the header row with [1:]

train = pd.read_csv("../input/train.csv")

test= pd.read_csv("../input/test.csv")

print('train datasets shape: ',train.shape)

print('test datasets shape: ',test.shape)

train.head()
test.head()
y_label = train["label"]

x_train = train.drop(labels = ["label"], axis =1)

del train

print('x train datasets shape: ',x_train.shape)

print('y label datasets shape: ',y_label.shape)
x_train = x_train/255.0

y_label = to_categorical(y_label, 10)

print('x train datasets shape: ',x_train.shape)

print('y label datasets shape: ',y_label.shape)

x_train.head()
x_train = x_train.values.reshape(-1, 28, 28, 1)

print('x train datasets shape: ',x_train.shape)
img = plt.imshow(x_train[10][:,:,0])
n = [10, 100,1000]

j = 0

for i in n:

    j+=1

    plt.subplot(310+(j))

    plt.imshow(x_train[i][:,:,0])

    plt.title("this is {}".format(y_label[i]))
model = Sequential()



model.add(Conv2D(filters = 16, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu', input_shape = (28,28,1)))

model.add(Conv2D(filters = 16, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))



print("model output shape: ", model.output_shape)



model.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(0.25))



print("model output shape: ", model.output_shape)



model.add(Flatten())



print("model output shape: ", model.output_shape)

model.add(Dense(64, activation = "relu"))

model.add(Dropout(0.5))

model.add(Dense(10, activation = "softmax"))



model.compile(

    optimizer='adam',

    loss='categorical_crossentropy',

    metrics=['accuracy']

)

lr_reduction = ReduceLROnPlateau(

    monitor='val_acc',

    patience=3,

    verbose=1,

    factor=0.5,

    min_lr=0.00001

)
history = model.fit(

    x_train,

    y_label,

    batch_size=1000,

    epochs=20,

    validation_split=0.2,

    verbose=2,

    callbacks=[lr_reduction]

)
history_dict = history.history

loss_values = history_dict['loss']

val_loss_values = history_dict['val_loss']

acc_values = history_dict['acc']

val_acc_values = history_dict['val_acc']

epochs = range(1,len(loss_values)+1)

plt.clf()

plt.subplot(311)

plt.plot(epochs, loss_values,'bo-')

plt.plot(epochs, acc_values,'rs-')

plt.xlabel('Iterations')

plt.ylabel('Loss & Accuracy ')

plt.title("For Train Data")



plt.subplot(313)

plt.plot(epochs, val_loss_values,'bo-')

plt.plot(epochs, val_acc_values,'rs-')

plt.xlabel('Iterations')

plt.ylabel('Loss & Accuracy')

plt.title("For validation Data")



plt.show()