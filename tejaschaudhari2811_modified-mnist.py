# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers import Dense, Conv2D, Flatten
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
import random
import mnist
%matplotlib inline
print("setup complete")
train_images = mnist.train_images() #import training data
test_images = mnist.test_images() # import test data
train_labels = mnist.train_labels() #import train labels
test_labels = mnist.test_labels() #import test labels
data = np.vstack((train_images,test_images)) #to merge all images together
labels = np.hstack((train_labels,test_labels)) # to merge all labels together
labels_1 = np.where(labels == 1)# to know where are the images with label 1 are present
alpha = labels_1[0][0:200:20] # to store random images of digit 1 in an array
print(alpha.shape)
#others  = labels[20:100]
#print(others.shape)
labels_3=np.where(labels==3)
labels_4=np.where(labels==4)
labels_5=np.where(labels==5)
labels_6=np.where(labels==6)
labels_7=np.where(labels==7)
labels_8=np.where(labels==8)
labels_9=np.where(labels==9)
c = labels_3[0][0:10]
d = labels_4[0][0:10]
e = labels_5[0][0:10]
f = labels_6[0][0:10]
g = labels_7[0][0:10]
h = labels_8[0][0:10]
i_ = labels_9[0][0:10]
labels_2 = np.where(labels == 2) # to know where the images with digit 2 are in data
beta = labels_2[0][0:1000:100] # to select random images with digit 2 as label in data
print(beta.shape)
labels_0 = np.where(labels == 0) #to know where the images with digit 0 are in data
null = labels_0[0][50:60] # to select random images with digit 0 as label in data
print(null.shape)
x = np.array([])
w = np.array([])
y=[]
z=[]
count = 0
for i in alpha:
    for j in alpha: 
        x = np.hstack((data[i],data[j]))
        w = np.vstack(np.dstack((labels[i],labels[j])))
        y.append(x)
        z.append(w)
for i in alpha:
    for j in beta: 
        x = np.hstack((data[i],data[j]))
        w = np.vstack(np.dstack((labels[i],labels[j])))
        y.append(x)
        z.append(w)
for i in alpha:
    for j in c: 
        x = np.hstack((data[i],data[j]))
        w = np.vstack(np.dstack((labels[i],labels[j])))
        y.append(x)
        z.append(w)
for i in alpha:
    for j in d: 
        x = np.hstack((data[i],data[j]))
        w = np.vstack(np.dstack((labels[i],labels[j])))
        y.append(x)
        z.append(w)
for i in alpha:
    for j in e: 
        x = np.hstack((data[i],data[j]))
        w = np.vstack(np.dstack((labels[i],labels[j])))
        y.append(x)
        z.append(w)
for i in alpha:
    for j in f: 
        x = np.hstack((data[i],data[j]))
        w = np.vstack(np.dstack((labels[i],labels[j])))
        y.append(x)
        z.append(w)
for i in alpha:
    for j in g: 
        x = np.hstack((data[i],data[j]))
        w = np.vstack(np.dstack((labels[i],labels[j])))
        y.append(x)
        z.append(w)
for i in alpha:
    for j in h: 
        x = np.hstack((data[i],data[j]))
        w = np.vstack(np.dstack((labels[i],labels[j])))
        y.append(x)
        z.append(w)
for i in alpha:
    for j in i_: 
        x = np.hstack((data[i],data[j]))
        w = np.vstack(np.dstack((labels[i],labels[j])))
        y.append(x)
        z.append(w)
for l in beta:
    for m in null:
        x = np.hstack((data[l],data[m]))
        w = np.vstack(np.dstack((labels[l],labels[m])))
        y.append(x)
        z.append(w)
dual_labels = np.array([11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20])
print(dual_labels.shape)
for k in range(700,703):
    first_image = np.array(y[k],dtype='float')
    #print(z[k])
    plt.imshow(first_image,cmap='gray')
    plt.show()
    # count += 1
      #  print(count)
print(type(y))
dual_mnist = np.array(y)
#print(type(z))
print(dual_mnist.shape)
dual_mnist_train,dual_mnist_test,dual_label_train,dual_label_test = train_test_split(dual_mnist,dual_labels,shuffle=True,random_state=1)
print(dual_mnist_train.shape,dual_mnist_test.shape,dual_label_train.shape ,dual_label_test.shape)
#print(dual_label_train)
dual_mnist_train = dual_mnist_train.reshape(750,28,56,1)
dual_mnist_test = dual_mnist_test.reshape(250,28,56,1)
print(dual_mnist_train.shape)
temp = dual_label_test
dual_label_train= to_categorical(dual_label_train)
dual_label_test = to_categorical(dual_label_test)
print(dual_label_train[0])
model = Sequential()
model.add(Conv2D(64,kernel_size=2, activation='relu',input_shape=(28,56,1)))
model.add(Conv2D(32, kernel_size= 2, activation='relu'))
model.add(Flatten())
model.add(Dense(21,activation='softmax'))
model.compile(
            optimizer = 'adam',
            loss = 'categorical_crossentropy',
            metrics =['accuracy'])

model.fit(dual_mnist_train, dual_label_train, validation_data=(dual_mnist_test, dual_label_test),
         epochs = 5)
model.evaluate(dual_mnist_test,dual_label_test)
preds=model.predict(dual_mnist_test[:])
print(np.argmax(preds, axis = 1))
print(temp[:])
