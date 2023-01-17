import numpy as np # numeric python

import pandas as pd # data management

import seaborn as sbn # data visualisation

import matplotlib.pyplot as pplt

%matplotlib inline

from keras.models import Sequential

from keras.layers.core import Dense,Dropout

from keras.layers import Conv2D, MaxPool2D, Flatten

from keras.regularizers import l2

from keras.utils import np_utils
# Print Data

# int: pos  indicates the position of item in the array

def printData(item):

    digit = item

    pplt.imshow(digit [:,:,0],cmap=pplt.cm.binary)

    pplt.show()

    

# Plots the Loss vs Accuracy Chart

# obj: history 

def plotChart(history):

    history_dict = history.history

    history_dict.keys()# look witch keys exist

    loss_values = history_dict['loss']

    acc_values = history_dict['accuracy']

    epochs = range(1, len(loss_values)+1)

    pplt.plot(epochs,loss_values,'r', label='training loss')

    pplt.plot(epochs,acc_values,'g',label='accuracy')

    pplt.title('loss and accuracy')

    pplt.xlabel('epochs')

    pplt.ylabel('loss vs. accuracy')

    pplt.legend()

    pplt.show

sample_submission = pd.read_csv("../input/digit-recognizer/sample_submission.csv")

test = pd.read_csv("../input/digit-recognizer/test.csv")

train = pd.read_csv("../input/digit-recognizer/train.csv")
train
# seperate the first column as Output-Data

YTrain = train["label"] 

# cut the 'label' column from training-data

XTrain = train.drop(labels = ["label"],axis = 1)  

del train

g = sbn.countplot(YTrain) 

YTrain.value_counts()

# normalisation to [0..1] instead of [0..255]

XTrain = XTrain / 255.0 

test = test / 255.0
# keras needs a shape (height,weight,channel)

# whith channel=1=greyscale, =3=rgb

XTrain = XTrain.values.reshape(-1,28,28,1) # -1 means all samples

print(XTrain.shape)

test = test.values.reshape(-1,28,28,1)

# the lables as "one-hot"-coding 

YTrain = np_utils.to_categorical(YTrain, num_classes = 10)

print(YTrain)

# like to see the first digit

printData(XTrain[0])

# like to see the last digit

printData(XTrain[41999])
# trying the CNN-model

CNN = Sequential()





CNN.add(Conv2D(32, kernel_size=(3, 3),

                 activation='relu',

                 input_shape=(28,28,1)))

CNN.add(Conv2D(64, (3, 3), activation='relu'))

CNN.add(MaxPool2D(pool_size=(2, 2)))

CNN.add(Dropout(0.25))

CNN.add(Flatten())

CNN.add(Dense(128, activation='relu'))

CNN.add(Dropout(0.5))

CNN.add(Dense(10, activation='softmax'))

# Summary

CNN.summary()
#Compiliation

CNN.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# Fitting the model

history=CNN.fit(XTrain,YTrain,epochs=5)
plotChart(history)
# prediction

results = CNN.predict(test) 

# index with the maximum probability

results = np.argmax(results,axis = 1) 

results = pd.Series(results,name="Label")
# Print test data

printData(test[100])

print("Prediction result:", results[100])





printData(test[220])

print("Prediction result:", results[220])