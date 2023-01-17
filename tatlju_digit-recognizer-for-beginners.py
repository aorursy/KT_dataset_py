# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn import svm
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
training_set = pd.read_csv('../input/digit-recognizer/train.csv')
#label_id pixel0 pixel 1 ...... pixel783
# 1       1        0             0
# 42000  0         1     ......   0
#print(training_set)
#print(training_set.shape)

test_set = pd.read_csv('../input/digit-recognizer/test.csv')
#print(test_set)
X_train = (training_set.iloc[:, 1:].values).astype('float32')
Y_train = training_set.iloc[:,:1].values.astype('int32') # labels
X_test = test_set.values.astype('float32')

images = training_set.iloc[:,1:]
labels = training_set.iloc[:,:1]



train_images, test_images,train_labels, test_labels = train_test_split(images, labels, train_size=0.8, random_state=0)
#Convert train datset to (num_images, img_rows, img_cols) format 

import matplotlib.pyplot as plt

X_train = X_train.reshape(X_train.shape[0], 28, 28)
for i in range(6, 9):
    plt.subplot(330 + (i+1))
    plt.imshow(X_train[i], cmap=plt.get_cmap('gray'))
    plt.title(Y_train[i]);
#expand 1 more dimention as 1 for colour channel gray
X_train = X_train.reshape(X_train.shape[0], 28, 28,1)
X_train.shape


#same situation but on a diffrent  way
######################################

b=6
for c in range(6,9):
    img=train_images.iloc[c].as_matrix()
    img=img.reshape((28,28))
    plt.subplot(330 +(c+1))
    plt.imshow(img,cmap='gray')
    plt.title(train_labels.iloc[c,0])
X_test = X_test.reshape(X_test.shape[0], 28, 28,1)
X_test.shape
##plt.hist braucht 2 dim von training_set ist es aber X_train anscheint nicht 
#print(training_set.iloc[7])
for y in range (6,9):
    plt.hist(training_set.iloc[y])
#clf = svm.SVC()
#clf.fit(train_images, train_labels.values.ravel())
#clf.score(test_images,test_labels)
test_images[test_images>0]=1
train_images[train_images>0]=1

#img=train_images.iloc[i].as_matrix().reshape((28,28))
#plt.imshow(img,cmap='binary')
#plt.title(train_labels.iloc[i])
for i in range(6,9):
    img=train_images.iloc[i].as_matrix().reshape((28,28))
    plt.subplot(330 + (i+1))
    plt.imshow(img, cmap='binary')
    plt.title(train_labels.iloc[i]);


for i in range(6,9):
    plt.hist(train_images.iloc[i])
clf = svm.SVC()
clf.fit(train_images, train_labels.values.ravel())
clf.score(test_images,test_labels)
#labeling the test data
#test_set[test_set>0]=1
#results=clf.predict(test_set[0:5000])
test_data=pd.read_csv('../input/digit-recognizer/test.csv')
test_data[test_data>0]=1
results=clf.predict(test_data[0:28000])
results


df = pd.DataFrame(results)
df.index.name='ImageId'
df.index+=1
df.columns=['Label']
df.to_csv('results.csv', header=True)
result_set = pd.read_csv('../input/digit-recognizer-for-beginners/results.csv')
print(result_set)