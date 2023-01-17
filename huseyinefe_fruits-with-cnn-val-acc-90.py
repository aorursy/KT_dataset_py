# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import cv2

from glob import glob

from keras.preprocessing.image import load_img,img_to_array

from sklearn.model_selection import train_test_split

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_images = []       

train_labels = []

shape = (128,128)  

train_path = '../input/fruit-images-for-object-detection/train_zip/train'

for filename in os.listdir('../input/fruit-images-for-object-detection/train_zip/train'):

    if filename.split('.')[1] == 'jpg':

        img = cv2.imread(os.path.join(train_path,filename))

        #0 is before 1 is after

        train_labels.append(filename.split('_')[0])

        img = cv2.resize(img,shape)

        train_images.append(img)

train_labels = pd.get_dummies(train_labels).values

train_images = np.array(train_images)

x_train,x_val,y_train,y_val = train_test_split(train_images,train_labels,random_state=1)

test_images = []

test_labels = []

shape = (128,128)

test_path = '../input/fruit-images-for-object-detection/test_zip/test'

for filename in os.listdir('../input/fruit-images-for-object-detection/test_zip/test'):

    if filename.split('.')[1] == 'jpg':

        img = cv2.imread(os.path.join(test_path,filename))

        test_labels.append(filename.split('_')[0])

        img = cv2.resize(img,shape)

        test_images.append(img)        

test_images = np.array(test_images)
print("x train shape :",x_train.shape)

print("x val shape :",x_val.shape)

print("y train shape :",y_train.shape)

print("y val shape :",y_val.shape)
img=x_train[40]

plt.imshow(img)

plt.title(y_train[40])

plt.axis("off")

plt.show()
test_img=test_images[25]

plt.imshow(test_img)

plt.title(test_labels[25])

plt.axis("off")

plt.show()
from keras.models import Sequential

from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D



model=Sequential()



model.add(Conv2D(filters=30,kernel_size=(3,3),padding="same",activation="relu",input_shape=(128,128,3)))

model.add(MaxPooling2D())

model.add(Conv2D(filters=30,kernel_size=(3,3),padding="same",activation="relu"))

model.add(MaxPooling2D())

model.add(Conv2D(filters=30,kernel_size=(3,3),padding="same",activation="relu"))

model.add(MaxPooling2D())

model.add(Conv2D(filters=30,kernel_size=(3,3),padding="same",activation="relu"))

model.add(MaxPooling2D())



model.add(Flatten())

model.add(Dense(206,activation="relu"))

model.add(Dropout(0.4))

model.add(Dense(103,activation="relu"))

model.add(Dropout(0.4))

model.add(Dense(4,activation="softmax"))
model.summary()
model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])
x_train=x_train/255.0

x_val=x_val/255.0
hist=model.fit(x_train,y_train,batch_size=30,epochs=100,validation_data=(x_val,y_val))
print(hist.history.keys())
plt.plot(hist.history["loss"],color="green",label="Train Loss")

plt.plot(hist.history["val_loss"],color="red",label="Validation Loss")

plt.title("Loss Plot")

plt.xlabel("Number of Epochs")

plt.ylabel("Loss Values")

plt.show()
plt.plot(hist.history["accuracy"],color="black",label="Train Accuracy")

plt.plot(hist.history["val_accuracy"],color="blue",label="Validation Accuracy")

plt.title("Accuracy Plot")

plt.xlabel("Number of Epochs")

plt.ylabel("Accuracy Values")

plt.show()
prediction=model.predict(x_val)
from copy import deepcopy

predicted_classes=deepcopy(prediction)

predicted_classes=np.argmax(predicted_classes,axis=1)

y_true=np.argmax(y_val,axis=1)

print("y predicted classes shape :",predicted_classes.shape)

print("y true shape :",y_true.shape)
from sklearn.metrics import confusion_matrix

cfm=confusion_matrix(y_true,predicted_classes)

f,ax=plt.subplots(figsize=(10,10))

sns.heatmap(cfm,annot=True,cmap="coolwarm",linecolor="black",linewidths=1,fmt=".0f",ax=ax)

plt.title(" Error Values of Validation with Heat Map")

plt.xlabel("Real Values")

plt.ylabel("Predicted Values")

plt.show()
train_prediction=model.predict(x_train)

train_predicted_classes=deepcopy(train_prediction)

train_predicted_classes=np.argmax(train_predicted_classes,axis=1)

train_true=np.argmax(y_train,axis=1)

cfm1=confusion_matrix(train_true,train_predicted_classes)

f,ax=plt.subplots(figsize=(10,10))

sns.heatmap(cfm1,annot=True,cmap="coolwarm",linecolor="black",linewidths=1,fmt=".0f",ax=ax)

plt.title(" Error Values of Train with Heat Map")

plt.xlabel("Real Values")

plt.ylabel("Predicted Values")

plt.show()