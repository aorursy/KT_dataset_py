# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import pydot

from sklearn import preprocessing 

from sklearn.model_selection import train_test_split



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

%matplotlib inline 



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")

output=pd.read_csv("../input/test.csv").values

y=train.label.values

X=train.drop(['label'],axis=1).values
#Visualisation of one image

sample_index = 8

plt.figure(figsize=(3, 3))

plt.imshow(X[sample_index].reshape(28,28), cmap=plt.cm.gray_r, interpolation='nearest')

plt.title("label: %d" % y[sample_index]);
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)



X_train_nor = preprocessing.scale(X_train)

X_test_nor = preprocessing.scale(X_test)

output_nor = preprocessing.scale(output)
from keras.utils.np_utils import to_categorical



Y_train = to_categorical(y_train)

Y_test = to_categorical(y_test)
from keras.models import Sequential

from keras.layers.core import Dense, Activation



N = X_train.shape[1]

H=128

K=10



model = Sequential()

model.add(Dense(H, input_dim=N,activation="relu"))

model.add(Dense(K, activation="softmax"))



model.summary()
from keras.optimizers import SGD

sgd = SGD(lr=0.1) 
model.compile(optimizer=sgd,loss='categorical_crossentropy',metrics=['accuracy'])
history = model.fit(X_train_nor, Y_train, epochs=10, batch_size=32,validation_split=0.1)
plt.figure(figsize=(10,5))



plt.plot(history.history['acc'],label='Train')

plt.plot(history.history['val_acc'],label='Validation')

plt.xlabel('# epochs')

plt.ylabel('Training loss')

#plt.ylim(0, 6)

plt.legend(loc='best');
model.evaluate(X_test_nor,Y_test)
y_predicted = model.predict_classes(X_test_nor)

errors = (y_predicted != y_test)



y_predicted_errors = y_predicted[errors]

y_test_errors = y_test[errors]

X_test_errors = X_test[errors]



plt.figure(figsize=(12, 9))

for i in range(15):

    plt.subplot(3, 5, i + 1)

    plt.imshow(X_test_errors[i].reshape(28, 28),cmap=plt.cm.gray_r, interpolation='nearest')

    plt.title("predicted label: %d\n true label: %d"% (y_predicted_errors[i], y_test_errors[i]))

  
X_train_nor = X_train_nor.reshape(-1,28,28,1)

X_test_nor = X_test_nor.reshape(-1,28,28,1)

output_nor = output_nor.reshape(-1,28,28,1)
from keras.preprocessing.image import ImageDataGenerator



datagen = ImageDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.1, # Randomly zoom image 

        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)

        horizontal_flip=False,  # randomly flip images

        vertical_flip=False)  # randomly flip images





datagen.fit(X_train_nor)
datagen.flow(X_train_nor,Y_train,batch_size=10)
from sklearn.metrics import confusion_matrix

from keras.layers import Dropout, Flatten, Conv2D, MaxPool2D



# Set the CNN model 

# my CNN architechture is In -> [[Conv2D->relu]*2 -> MaxPool2D -> Dropout]*2 -> Flatten -> Dense -> Dropout -> Out



model = Sequential()



model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', activation ='relu', input_shape = (28,28,1)))

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))





model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(0.25))





model.add(Flatten())

model.add(Dense(256, activation = "relu"))

model.add(Dropout(0.5))

model.add(Dense(10, activation = "softmax"))
model.compile(optimizer = sgd , loss = "categorical_crossentropy", metrics=["accuracy"])
history = model.fit_generator(datagen.flow(X_train_nor,Y_train,batch_size=64),

                              epochs=5,

                              steps_per_epoch = X_train_nor.shape[0] // 64 , 

                              validation_data = (X_test_nor,Y_test))

model.evaluate(X_test_nor,Y_test)
y_predicted = model.predict_classes(X_test_nor)

errors = (y_predicted != y_test)



y_predicted_errors = y_predicted[errors]

y_test_errors = y_test[errors]

X_test_errors = X_test[errors]



plt.figure(figsize=(24, 18))

for i in range(30):

    plt.subplot(6, 5, i + 1)

    plt.imshow(X_test_errors[i].reshape(28, 28),cmap=plt.cm.gray_r, interpolation='nearest')

    plt.title("predicted label: %d\n true label: %d"% (y_predicted_errors[i], y_test_errors[i]))

    plt.xticks([])

  

  
y_output = model.predict_classes(output_nor)

df_output = pd.DataFrame({"ImageId" : np.arange(1,len(y_output)+1),"Label": y_output})

df_output.set_index("ImageId", inplace=True)

df_output.to_csv("prediction3.csv")