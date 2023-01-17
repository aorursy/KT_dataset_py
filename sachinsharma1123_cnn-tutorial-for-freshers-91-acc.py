# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#lets make a list of categories.ie infected or non infected leaf

categories=list(os.listdir('/kaggle/input/corn-leaf-infection-dataset/Corn Disease detection'))
#now preprocess the data from the directory

import matplotlib.pyplot as plt

import seaborn as sns

import cv2

dire='/kaggle/input/corn-leaf-infection-dataset/Corn Disease detection'

features=[]

IMG_SIZE=200

for i in categories:

    path=os.path.join(dire,i)

    num_classes=categories.index(i)

    for img in os.listdir(path):

        if img.endswith('.jpg'):

            

            img_array=cv2.imread(os.path.join(path,img))

            img_array=cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))

            features.append([img_array,num_classes])

#now separate the dependent and independent variables from the list

X=[]

Y=[]

for i ,j in features:

    X.append(i)

    Y.append(j)
#lets visualize the data before normalising

for i in range(1,5):

    plt.imshow(X[i])

    plt.xlabel(Y[0])

    plt.show()
#lets normalize the training data

x=np.array(X)/255
#after normalising

for i in range(1,5):

    plt.imshow(x[i])

    plt.xlabel(Y[0])

    plt.show()
#lets reshape the x array to meet the keras requirement

x=x.reshape(-1,200,200,3)
x.shape
#we need to convert target lables into one hot encoding integers

from tensorflow.keras.utils import to_categorical

y=to_categorical(Y)
#now we have to split our data into train and test sets

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=12,test_size=0.2)

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import MaxPool2D,Flatten,Dense,BatchNormalization,Dropout,Conv2D

model=Sequential([

    Conv2D(64,(3,3),padding='same',activation='relu',input_shape=(200,200,3)),

    MaxPool2D((2,2)),

    Conv2D(128,(3,3),activation='relu'),

    MaxPool2D((3,3)),

    Dropout(0.2),

    BatchNormalization(),

    Conv2D(256,(3,3),padding='same',activation='relu'),

    MaxPool2D((3,3)),

    Conv2D(512,(3,3),activation='relu',padding='same'),

    MaxPool2D((2,2)),

    Dropout(0.3),

    BatchNormalization(),

    Flatten(),

    Dense(1024,activation='relu'),

    Dense(2,activation='sigmoid')

])
model.summary()
model.compile(optimizer='Adam',loss='mae',metrics=['acc'])
history=model.fit(x_train,y_train,epochs=40,batch_size=56,validation_split=0.3)
loss,accuracy=model.evaluate(x_test,y_test)
#plot the model results to evaluate better

plt.figure(figsize=(12,5))

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.xlabel('epochs')

plt.ylabel('accuracy score')

plt.title('accuracy score vs epochs')

plt.legend(['train','test'])

plt.show()
#there are many sharp peaks during the training phase but at the end it achieves a quite good accuracy 91 on test set
#plot the loss score of both test and train sets

plt.figure(figsize=(12,5))

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.xlabel('epochs')

plt.ylabel('loss')

plt.title('loss vs epochs')

plt.legend(['train','test'])

plt.show()