# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# visualization libraries

import seaborn as sns 

import matplotlib.pyplot as plt

%pylab inline

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory





from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix



import itertools



import keras

from keras import models

from keras import layers

from keras.models import Sequential

from keras.preprocessing.image import ImageDataGenerator

from keras.layers import Dense, BatchNormalization, Conv2D, MaxPool2D, Flatten, Dropout

from keras.optimizers import Adam

from keras.utils.np_utils import to_categorical



#declaring variables

input_shape = (28, 28, 1)

num_classes = 10

epochs = 20 

batch_size = 64



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#load training and testing datasets

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
train.head() #first 5 rows of training dataset
test.head() # first 5 rows of testing dataset
X_train= train.iloc[:,1:]

y_train = train['label']

X_train = X_train.values.astype('float32')

y_train = y_train.values.astype('int32')

X_test_df = test.values.astype('float32')
# To count the values of each class

sns.countplot(y_train)
#reshae the data in between 0 and 1

X_test_df = X_test_df/255

X_train=X_train/255.



#reshape image in 3_dimension

X_train=X_train.reshape(-1,28,28,1)

X_test_df = X_test_df.reshape(-1,28,28,1)
X_train.shape
y_train.shape
X_test_df.shape
plt.figure(figsize=(15,9))

for i in range(40):

    plt.subplot(4,10,i+1)

    plt.imshow(X_train[i].reshape(28,28),cmap='gray')
#one-hot-encoding

y_train= to_categorical(y_train,num_classes = num_classes)
#we use train_test_split for training and testing data

X_train, X_test, y_train, y_test = train_test_split(X_train,y_train,test_size=.2)
X_train.shape
X_test.shape
y_train.shape
y_test.shape
# Create Neural Network model, onjects of model and then add layers

# In this model we used 3 hiddden layers, 1 input and 1 output

# We used "ReLu- Rectified Linear units" as it is simpler and efficient when we used hiddn layers



model = models.Sequential()



model.add(layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = input_shape))

model.add(layers.MaxPooling2D(2, 2))

model.add(layers.Dropout(0.5))



model.add(layers.Conv2D(64, (3, 3), activation = 'relu'))

model.add(layers.MaxPooling2D(2, 2))

model.add(layers.Dropout(0.5))



model.add(layers.Conv2D(64, (3, 3), activation = 'relu'))

model.add(layers.Flatten())



model.add(layers.Dense(128, activation = 'relu'))

model.add(layers.Dropout(0.5))



# For output layer we used "softmax" function,to compute the probabilities for each classes  

model.add(layers.Dense(10, activation = 'softmax')) 
model.summary()
# Compile it

model.compile(loss = keras.losses.categorical_crossentropy, metrics=['accuracy'],optimizer='rmsprop')
# Now we are ready to start training phase

model.fit(X_train,y_train,epochs = epochs, batch_size = batch_size,validation_data=(X_test, y_test), verbose=2)
# Testing phase

final_loss, final_accuracy = model.evaluate(X_test,y_test,verbose=0)



#printing the final loss and final accuracy of our model

print("final_loss: {0:.6f}, final_accuracy: {1:.6f}".format(final_loss, final_accuracy))
#Test your data against trained model to find out predictons

test_prediction = pd.DataFrame(model.predict(X_test_df, batch_size=150))

test_prediction = pd.DataFrame(test_prediction.idxmax(axis=1))

test_prediction.index.name = 'ImageId'

test_prediction = test_prediction.rename(columns={0:'Label'}).reset_index()

test_prediction['ImageId'] = test_prediction['ImageId']+1

test_prediction.head(10)
#Generating CSV file of predictions

fileName = 'MNIST_Predictions1.csv'

test_prediction.to_csv(fileName,index=False)

print('Saved File Name'+fileName)