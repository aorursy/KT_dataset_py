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
df=pd.read_csv("../input/drug200.csv", delimiter=",")
df.dropna
df.head()
from sklearn.tree import DecisionTreeClassifier

X=df[["Age","Sex","BP","Cholesterol","Na_to_K"]].values

X
from sklearn import preprocessing

le_sex=preprocessing.LabelEncoder()

le_sex.fit(['F','M'])

X[:,1]=le_sex.transform(X[:,1])





le_BP = preprocessing.LabelEncoder()

le_BP.fit([ 'LOW', 'NORMAL', 'HIGH'])

X[:,2] = le_BP.transform(X[:,2])





le_Chol = preprocessing.LabelEncoder()

le_Chol.fit([ 'NORMAL', 'HIGH'])

X[:,3] = le_Chol.transform(X[:,3]) 



X[:5]
y=df["Drug"]



from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=3)
print(x_train.shape)

print(y_train.shape)

print(x_test.shape)

print(y_test.shape)


drugTree=DecisionTreeClassifier(criterion="entropy",max_depth=4)
drugTree.fit(x_train,y_train)
y_predict=drugTree.predict(x_test)

y_predict
from sklearn import metrics

import matplotlib.pyplot as plt

metrics.accuracy_score(y_test,y_predict)