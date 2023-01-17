# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#import libraries

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

import itertools

from keras.models import Sequential

from keras.layers import Dense, Dropout,Flatten,Conv2D,MaxPool2D

from keras.optimizers import RMSprop, Adam

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau

#load dataset

#dataset overview

#Now  we have 2 numpy files X.npy and Y.npy. So we need to load them.



dataset1=np.load('X.npy')

print(dataset1)



dataset2=np.load('Y.npy')

print(dataset2)

#Determine to variables

imageSize=64

#We can shown as in the figure.For example we want to see what is in the 12. data. 

#If you run it, you can see "3" digit data with the fingers of hand in the 12. data.

plt.imshow(dataset1[12].reshape(imageSize,imageSize))

plt.title("12. image")

#Now, we do not want to see the with axises.(Optional)

plt.axis("off")

#Now if we see the X.npy and Y.npy dataset. X.npy dataset (2062,64,64), Y.npy dataset is (2062,10).

#For 0: [0:204]

#For 1: [204:409]

#For 2:[409:615]

#♣For 3:[615:822]

#◘For 4:[822:1028]

#For 5: [1028:1236]

#For 6:[1236:1443]

#For 7:[1443:1649]

#For 8:[1649:1855]

#For 9[1855:2062]

#dataset2=dataset2.reshape(2062,1)

print("Dimension of Dataset1:" + str(dataset1.ndim))

print("Dataset1 shape:" +str( dataset1.shape))

print("Dataset2 shape:"+ str(dataset2.shape))
#Train and Test Split

#training and testing data

from sklearn.model_selection import train_test_split

Xtrain, Xtest, Ytrain, Ytest=train_test_split(dataset1,dataset2, test_size=0.15, random_state=42)

trainNumber=Xtrain.shape[0]

testNumber=Ytest.shape[0]

trainNumber2=Ytrain.shape[0]

testNumber2=Xtest.shape[0]

print("Xtrain Number: " + str(trainNumber) + "Xtrain shape: " + str(Xtrain.shape))

print("Ytest Number:" + str(testNumber) + "Ytest shape: " + str(Ytest.shape))

print("Ytrain Number:" + str(trainNumber2) + "Ytrain shape: " + str(Ytrain.shape))

print("Xtest Number:" + str(testNumber2) + "Xtest shape: " + str(Xtest.shape))
#Normalizationing Data; convert to gray scale: between 0 and 1

print("Normalization")

Xtrain=Xtrain/255.0

Xtest=Xtest/255.0

Ytrain=Ytrain/255.0

Ytest=Ytest/255.0

#Reshape Data because of the Keras rules

#-1?

Xtrain=Xtrain.reshape(-1,64,64,1)

Xtest=Xtest.reshape(-1,64,64,1)

#Ytrain=Ytrain.reshape(-1,10,1)

print("Xtrain shape: " , Xtrain.shape)

print("Xtest shape: ",Xtest.shape)

print("Ytest shape: ",Ytest.shape)

#Label Encoding

from keras.utils.np_utils import to_categorical

#Ytrain=to_categorical(Ytrain,num_classes=10)

#Create a model



learningModel=Sequential()

#CNN-Same Padding

learningModel.add(Conv2D(filters=8,kernel_size=(5,5),padding="Same",activation='relu',input_shape=(64,64,1)))

#CNN-Max Pooling

learningModel.add(MaxPool2D(pool_size=(2,2)))

#Flattening

learningModel.add(Dropout(0.25))

#CNN-Same Padding2

learningModel.add(Conv2D(filters=2,kernel_size=(3,3),padding="Same", activation="relu"))

#CNN-Max Pooling2

learningModel.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

#Flattening2

learningModel.add(Dropout(0.25))

#Full Connection

learningModel.add(Flatten())

learningModel.add(Dense(256,activation="relu"))

learningModel.add(Dropout(0.5))

learningModel.add(Dense(10,activation="softmax"))

#Define Optimizer

learningOptimizer=Adam(lr=0.001,beta_1=0.9,beta_2=0.999)

#Compile Model

learningModel.compile(optimizer=learningOptimizer,loss="categorical_crossentropy",metrics=["accuracy"])

#Epochs and Batch Size

epoch=1

batchSize=1

#Data Augmentation

dataGeneration=ImageDataGenerator(featurewise_center=False,samplewise_center=False,

                                  featurewise_std_normalization=False, samplewise_std_normalization=False,

                                  zca_whitening=False,rotation_range=0.5,zoom_range=0.5,width_shift_range=0.5,

                                  height_shift_range=0.5,horizontal_flip=False,vertical_flip=False)

dataGeneration.fit(Xtrain)



print("**********************")

print(len(Xtrain))

print(Xtrain.shape)



#Fit the Model

"""history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size),

                              epochs = epochs, validation_data = (X_val,Y_val), steps_per_epoch=X_train.shape[0] // batch_size)

"""

modelProcess=learningModel.fit_generator(dataGeneration.flow(Xtrain,Ytrain,batch_size=batchSize),

                                         epochs=epoch,validation_data = (Xtest,Ytest),

                                         steps_per_epoch=Xtrain.shape[0])

#Evaluate the Model



plt.plot(modelProcess.history['val_loss'],color="b", label="Validation Loss")

plt.title("Test Loss")

plt.xlabel("Number Of Epochs")

plt.ylabel("Loss")

plt.legend()

plt.show()

#Evaluate the Model with Confusion Matrix

import seaborn as sns

#Predict the values

Yprediction=learningModel.predict(Xtest)

#Convert the prediction 

YpredictionClasses=np.argmax(Yprediction,axis=1)

#Convert the true predictions

Ytrue=np.argmax(Ytest,axis=1)

#Confusion Matrix

confusionMatrix=confusion_matrix(Ytrue,YpredictionClasses)

#Show the results

ref, ax=plt.subplots(figsize=(8,8))

sns.heatmap(confusionMatrix,annot=True,linewidths=0.01,cmap="Greens",linecolor="gray",fmt=".1f", ax=ax)

plt.xlabel("Predicted Label")

plt.ylabel("True Label")

plt.title("Confusion Matrix")

plt.show()
import pandas as pd

sample_submission = pd.read_csv("../input/digit-recognizer/sample_submission.csv")

test = pd.read_csv("../input/digit-recognizer/test.csv")

train = pd.read_csv("../input/digit-recognizer/train.csv")
import pandas as pd

sample_submission = pd.read_csv("../input/digit-recognizer/sample_submission.csv")

test = pd.read_csv("../input/digit-recognizer/test.csv")

train = pd.read_csv("../input/digit-recognizer/train.csv")