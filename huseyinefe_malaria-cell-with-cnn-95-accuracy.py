import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import cv2

from copy import deepcopy

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix,roc_auc_score,roc_curve

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
uninfected_images = []       

shape = (100,100)  

uninfected_path = '/kaggle/input/cell-images-for-detecting-malaria/cell_images/Uninfected'

for filename in os.listdir('/kaggle/input/cell-images-for-detecting-malaria/cell_images/Uninfected'):

    if filename.split('.')[1] == 'png':

        img = cv2.imread(os.path.join(uninfected_path,filename))

        img = cv2.resize(img,shape)

        uninfected_images.append(img)

uninfected_images = np.array(uninfected_images)

parasitized_images = []

shape = (100,100)

parasitized_path = '/kaggle/input/cell-images-for-detecting-malaria/cell_images/Parasitized'

for filename in os.listdir('/kaggle/input/cell-images-for-detecting-malaria/cell_images/Parasitized'):

    if filename.split('.')[1] == 'png':

        img = cv2.imread(os.path.join(parasitized_path,filename))

        img = cv2.resize(img,shape)

        parasitized_images.append(img)        

parasitized_images = np.array(parasitized_images)
print("Uninfected Images shape :",uninfected_images.shape)

print("Parasitized Images shape :",parasitized_images.shape)
#uninfected_images = 0

#parasitized_images = 1

uninfected_numbers = np.zeros(uninfected_images.shape[0])

parasitized_numbers = np.ones(parasitized_images.shape[0])

print("Uninfected Numbers shape :",uninfected_numbers.shape)

print("Parasitized Numbers shape :",parasitized_numbers.shape)
x=np.concatenate((uninfected_images,parasitized_images),axis=0)

y=np.concatenate((uninfected_numbers,parasitized_numbers),axis=0)

y=y.reshape(y.shape[0],1)

print("x shape :",x.shape)

print("y shape :",y.shape)
img=x[48]

plt.imshow(img)

plt.title(y[48])

plt.axis("off")

plt.show()
sns.countplot(y.reshape(y.shape[0], ))

plt.legend()

plt.title("Count of Labels")

plt.xlabel("Different Labels")

plt.ylabel("Count")

plt.show()
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=42)

print("x train shape :",x_train.shape)

print("x test shape :",x_test.shape)

print("y train shape :",y_train.shape)

print("y test shape :",y_test.shape)
model=Sequential()



model.add(Conv2D(filters=30,kernel_size=(3,3),activation="relu",padding="same",input_shape=(100,100,3)))

model.add(MaxPooling2D())

model.add(Dropout(0.4))

model.add(Conv2D(filters=30,kernel_size=(3,3),activation="relu",padding="same"))

model.add(MaxPooling2D())

model.add(Dropout(0.4))



model.add(Flatten())

model.add(Dense(206,activation="relu"))

model.add(Dropout(0.4))

model.add(Dense(103,activation="relu"))

model.add(Dropout(0.4))

model.add(Dense(1,activation="sigmoid"))
model.summary()
model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
hist=model.fit(x_train,y_train,batch_size=250,epochs=100,validation_data=(x_test,y_test))
print(hist.history.keys())
train_loss=hist.history["loss"]

test_loss=hist.history["val_loss"]

train_accuracy=hist.history["accuracy"]

test_accuracy=hist.history["val_accuracy"]
plt.plot(train_loss[1:],color="red",label="Train Loss")

plt.plot(test_loss[1:],color="green",label="Test Loss")

plt.title("Loss Plot")

plt.legend()

plt.xlabel("Number of Epochs")

plt.ylabel("Loss Values")

plt.grid()

plt.show()
plt.plot(train_accuracy,color="black",label="Train Accuracy")

plt.plot(test_accuracy,color="blue",label="Test Accuracy")

plt.title("Accuracy Plot")

plt.legend()

plt.xlabel("Number of Epochs")

plt.ylabel("Accuracy Values")

plt.grid()

plt.show()
prediction=model.predict(x_test)

predicted_classes=deepcopy(prediction)

for i in range(0,predicted_classes.shape[0]):

    if predicted_classes[i]>0.5:

        predicted_classes[i]=1

    else:

        predicted_classes[i]=0
print("predicted classes shape :",predicted_classes.shape)

print("y test :",y_test.shape)
cfm=confusion_matrix(y_test,predicted_classes)

f,ax=plt.subplots(figsize=(10,10))

sns.heatmap(cfm,annot=True,cmap="coolwarm",linecolor="black",linewidths=1,fmt=".0f",ax=ax)

plt.title("Error Values with Heat Map")

plt.xlabel("Real Values")

plt.ylabel("Predicted Values")

plt.show()
fpr,tpr,thresholds=roc_curve(y_test,prediction)

print("FPR shape :",fpr.shape)

print("TPR shape :",tpr.shape)
plt.plot(fpr,color="red",label="FPR")

plt.plot(tpr,color="green",label="TPR")

plt.legend()

plt.title("Roc Curve Plot")

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.show()
auc_score=roc_auc_score(y_test,prediction)

print("Roc Auc Score Values :",auc_score)