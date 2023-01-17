# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import tensorflow as tf

from sklearn import svm

from sklearn.metrics import accuracy_score

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
Class=pd.read_csv("/kaggle/input/zoo-animal-classification/class.csv")

Zoo=pd.read_csv("/kaggle/input/zoo-animal-classification/zoo.csv")
Class
Zoo
plt.bar(Class['Class_Type'],Class['Number_Of_Animal_Species_In_Class'])

plt.xlabel('Class_Type')

plt.ylabel('Number_Of_Animal_Species_In_Class')
Train_Data=Zoo.iloc[0:75,1:18]

Test_Data=Zoo.iloc[75:,1:18]

x_train=Train_Data.iloc[:,0:16]

y_train=Train_Data.iloc[:,16:17]

x_test=Test_Data.iloc[:,0:16]

y_test=Test_Data.iloc[:,16:17]
X_TRAIN=np.asarray(x_train)

X_TEST=np.asarray(x_test)

Y_TRAIN=np.asarray(y_train)

Y_TEST=np.asarray(y_test)

model=tf.keras.models.Sequential()

model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(64,activation=tf.nn.relu))

model.add(tf.keras.layers.Dense(8,activation=tf.nn.softmax))



model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

loss=model.fit(X_TRAIN,Y_TRAIN,epochs=10)
val_loss,val_acc=model.evaluate(X_TEST,Y_TEST)
clf1=svm.SVC(gamma=1,C=1).fit(X_TRAIN,Y_TRAIN)

s=clf1.predict(X_TEST)

print(accuracy_score(Y_TEST,s))

SVM_ACC=clf1.score(X_TEST,Y_TEST)

clf2=GaussianNB().fit(X_TRAIN,Y_TRAIN)

s2=clf2.predict(X_TEST)

print(accuracy_score(Y_TEST,s2))

GNB_ACC=clf2.score(X_TEST,Y_TEST)
clf3=KNeighborsClassifier(n_neighbors=3).fit(X_TRAIN,Y_TRAIN)

s3=clf3.predict(X_TEST)

print(accuracy_score(Y_TEST,s2))

KNN_ACC=clf3.score(X_TEST,Y_TEST)
Y=[val_acc*100,SVM_ACC*100,GNB_ACC*100,KNN_ACC*100]

X=['Neural Net','SVM','Naive Bayes','KNN']



plt.bar(X,Y)