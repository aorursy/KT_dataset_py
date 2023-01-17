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
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets
plt.style.use('seaborn')
iris=pd.read_csv('../input/Iris.csv')
iris.head()
iris.info()
iris.drop('Id', axis=1,inplace=True)
iris.head()
iris['Species'].value_counts()
plt.figure(figsize = (15,10))
plt.subplot(2,2,1)
sns.violinplot(x='Species',y='SepalLengthCm',data=iris)
plt.subplot(2,2,2)
sns.violinplot(x='Species',y='PetalLengthCm',data=iris)
plt.subplot(2,2,3)
sns.violinplot(x='Species',y='SepalWidthCm',data=iris)
plt.subplot(2,2,4)
sns.violinplot(x='Species',y='PetalWidthCm',data=iris)


sns.pairplot(iris,hue='Species')
sns.set
fig = plt.gcf()
fig.set_size_inches(8,7)
fig = sns.swarmplot(x="Species", y="PetalLengthCm", data=iris)

from sklearn.model_selection import train_test_split
train, test = train_test_split(iris, test_size=.22, random_state=0)
print(train.shape)
print(test.shape)
train_X = train[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
train_y = train.Species
test_X= test[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']] 
test_y = test.Species
train_X.head()

train_y.head()
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
model = SVC()
model.fit(train_X,train_y) 
prediction=model.predict(test_X) 
acc_svc = round(accuracy_score(prediction,test_y) * 100, 2)
print(acc_svc)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
model = LogisticRegression()
model.fit(train_X,train_y) 
prediction=model.predict(test_X) 
acc_svc = round(accuracy_score(prediction,test_y) * 100, 2)
print(acc_svc)
from sklearn.ensemble import GradientBoostingClassifier

gbk = GradientBoostingClassifier()
gbk.fit(train_X, train_y)
y_pred = gbk.predict(test_X)
acc_gbk = round(accuracy_score(y_pred,test_y) * 100, 2)
print(acc_gbk)
from sklearn.ensemble import RandomForestClassifier
randomforest = RandomForestClassifier()
randomforest.fit(train_X, train_y)
y_pred = randomforest.predict(test_X)
acc_randomforest = round(accuracy_score(y_pred,test_y)*100, 2)
print(acc_randomforest)