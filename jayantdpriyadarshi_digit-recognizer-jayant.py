# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train=pd.read_csv("../input/train.csv")
test=pd.read_csv("../input/test.csv")
train.head()
train1,test1=train_test_split(train,test_size=.3,random_state=100)
##prepration of data 
train1_x=train1.drop("label",axis=1)
train1_y=train1['label']
test1_x= test1.drop("label",axis=1)
test1_y= test1["label"]
print(train1_x.shape)
print(train1_y.shape)
print(test1_x.shape)
print(test1_y.shape)
from sklearn.tree import DecisionTreeClassifier
model_dt=DecisionTreeClassifier(random_state=100)
model_dt.fit(train1_x,train1_y)
pred_dt = model_dt.predict(test1_x)

accuracy_dt = accuracy_score(pred_dt,test1_y)
print(accuracy_dt)
model_rf=RandomForestClassifier(random_state=100,n_estimators=1000)
model_rf.fit(train1_x,train1_y)
pred_rf= model_rf.predict(test1_x)

accuracy_rf = accuracy_score(pred_rf,test1_y)
print(accuracy_rf)
model_ab=AdaBoostClassifier(random_state=100)
model_ab.fit(train1_x,train1_y)
pred_ab= model_ab.predict(test1_x)

accuracy_ab = accuracy_score(pred_ab, test1_y)
print(accuracy_ab)

model_knn = KNeighborsClassifier(n_neighbors=7)
model_knn.fit(train1_x,train1_y)
pred_knn= model_knn.predict(test1_x)

accuracy_knn = accuracy_score(pred_knn, test1_y)
print(accuracy_knn)
### working on original datasets

train_x=train.drop("label",axis=1)
train_y=train["label"]
model=RandomForestClassifier(random_state=100,n_estimators=1000)
model.fit(train_x,train_y)
test_predict= model.predict(test)
dt_predict=pd.DataFrame({"Label":test_predict})
dt_predict["ImageId"]=test.index+1
dt_predict[["ImageId","Label"]].to_csv("prediction.csv",index=False)
