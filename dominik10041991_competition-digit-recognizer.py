# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import re
import matplotlib.pyplot as plt, matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn import metrics
%matplotlib inline

labeled_images = pd.read_csv('../input/train.csv')
images = labeled_images.iloc[0:10000,1:]
labels = labeled_images.iloc[0:10000,:1]
train_images, test_images,train_labels, test_labels = train_test_split(images, labels, train_size=0.8, random_state=0)
clf=RandomForestClassifier(n_estimators=1500, max_depth=25,random_state=1)
clf.fit(train_images, train_labels.values.ravel())
clf.score(test_images,test_labels)
clf = svm.SVC(kernel='rbf', C=7, gamma=0.009)
clf.fit(train_images, train_labels.values.ravel())
clf.score(test_images,test_labels)
#Teste Support Vektor Machine
clf = svm.SVC(kernel='linear')
clf.fit(train_images, train_labels.values.ravel())
clf.score(test_images,test_labels)
#Binarisiere die Bilder
test_images[test_images>0]=1
train_images[train_images>0]=1
#Teste Binarisierte Support Vektor Machine
clf = svm.SVC(kernel='rbf', C=7, gamma=0.009)
clf.fit(train_images, train_labels.values.ravel())
clf.score(test_images,test_labels)
#Teste Binarisierte Random Forest Classifier
clf=RandomForestClassifier(n_estimators=1500, max_depth=25,random_state=1)
clf.fit(train_images, train_labels.values.ravel())
clf.score(test_images,test_labels)
#Teste Binarisierte Support Vektor Machine
clf = svm.SVC(kernel='rbf', C=7, gamma=0.009)
clf.fit(train_images, train_labels.values.ravel())
clf.score(test_images,test_labels)
test_data=pd.read_csv('../input/test.csv')
test_data[test_data>0]=1
results=clf.predict(test_data[0:28000])
print (results)
df = pd.DataFrame(results)
df.index+=1
df.index.name='ImageId'
df.columns=['Label']
df.to_csv('results20181019.csv', header=True)
