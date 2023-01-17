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
data=pd.read_csv("../input/heart.csv")

data.head()
data.shape
data_=data.iloc[:,0:13]

target=data.iloc[:,13:14]

data_.shape,target.shape
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(data_,target)
from sklearn.linear_model import LogisticRegression

clf1=LogisticRegression()

clf1.fit(x_train,y_train)
y_pred1=clf1.predict(x_test)
print(clf1.score(x_train,y_train))

print(clf1.score(x_test,y_test))
from sklearn.tree import DecisionTreeClassifier

clf2=DecisionTreeClassifier()
clf2.fit(x_train,y_train)
y2_pred=clf2.predict(x_test)
print(clf2.score(x_train,y_train))

print(clf2.score(x_test,y_test))