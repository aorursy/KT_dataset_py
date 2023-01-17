# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import glob
import cv2

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import warnings
# filter warnings
warnings.filterwarnings('ignore')

import os
#print(os.listdir("../input"))
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
# Any results you write to the current directory are saved as output.

# Any results you write to the current directory are saved as output.
("../input/fruits-360_dataset/fruits-360/Training/Banana/*.JPG")

fruits=[]
files=glob.glob("../input/fruits-360_dataset/fruits-360/Training/Banana/*")
files2=glob.glob("../input/fruits-360_dataset/fruits-360/Training/Avocado/*")

for i in files:
    im=cv2.imread(i,0)
    fruits.append(im)
for i in files2:
    im2=cv2.imread(i,0)
    fruits.append(im2)
    
fruits2=np.asarray(fruits)
fruits2=fruits2/255
x=fruits2
zeros=np.zeros(427)
ones=np.ones(490)
y=np.concatenate((zeros,ones),axis=0).reshape(x.shape[0],1)
    
print(x.shape)
print(y.shape)
plt.subplot(1,2,1)
plt.imshow(x[260])
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(x[500])
plt.axis("off")
plt.show()
plt.subplot(1,2,1)
plt.imshow(x[1])
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(x[900])
plt.axis("off")
plt.show()
plt.subplot(1,2,1)
plt.imshow(x[120])
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(x[820])
plt.axis("off")
plt.show()
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
x_train_flatten=x_train.reshape(x_train.shape[0],x_train.shape[1]*x_train.shape[2])
x_test_flatten=x_test.reshape(x_test.shape[0],x_test.shape[1]*x_train.shape[2])
print(x_train_flatten.shape)
print(x_test_flatten.shape)
def sigmoid(z):
    y_head=1/(1+np.exp(-z))
    return y_head
x_train_flatten.shape
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import  Dense

def build_classifier():
    classifier=Sequential()
    classifier.add(Dense(units=8,kernel_initializer="uniform",activation="relu",
                         input_dim=x_train_flatten.shape[1]))
    classifier.add(Dense(units=4,kernel_initializer="uniform",activation="relu"))
    classifier.add(Dense(units=1,kernel_initializer="uniform",activation="sigmoid"))
    classifier.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
    
    return classifier
classifier=KerasClassifier(build_fn=build_classifier,epochs=10)
accuracies=cross_val_score(estimator=classifier,X=x_train_flatten,y=y_train,cv=3)
mean=accuracies.mean()
variance=accuracies.std()

print("accuracy mean: ",str(mean))
print("accuracy variance: ",str(variance))

import seaborn as sns
from sklearn.metrics import confusion_matrix
classifier.fit(x_train_flatten,y_train)
y_pred=classifier.predict(x_train_flatten)
conf_mat=confusion_matrix(y_train,y_pred)
f,ax=plt.subplots(figsize=(5,5))
sns.heatmap(conf_mat,annot=True,linewidths=0.01,cmap="Greens",linecolor="gray",fmt=".1f",ax=ax)
plt.show()

