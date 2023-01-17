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
#importing dataset
import pandas as pd
iris = pd.read_csv('../input/Iris.csv')
iris.head(5)
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import svm
from sklearn import metrics
#training 70% test 30%
train, test = train_test_split(iris, test_size = .3)
print(train.shape)
print(test.shape)
iris.head(2)
trainX = train[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
trainY = train.Species

testX = test[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
testY = test.Species
#Logistic Regression
model = LogisticRegression()
model.fit(trainX,trainY)
prediction = model.predict(testX)
print('Accuracy is :',metrics.accuracy_score(prediction,testY))

#Support Vector Machine
model = svm.SVC()
model.fit(trainX, trainY)
prediction = model.predict(testX)
print('Accuracy is :',metrics.accuracy_score(prediction,testY))
