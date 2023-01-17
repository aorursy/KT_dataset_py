# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

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
train1,test1 = train_test_split(train,test_size=0.3,random_state=100)



train1_x = train1.drop('label',axis = 1)

train1_y = train1['label']

test1_x =  test1.drop('label',axis = 1)

test1_y = test1['label']
model_dt = DecisionTreeClassifier(random_state=100,max_depth=10)

model_dt.fit(train1_x,train1_y)

pred = model_dt.predict(test1_x)

accuracy_dt=accuracy_score(pred,test1_y)

print(accuracy_dt)
model_rf = RandomForestClassifier(random_state=100,n_estimators=300)

model_rf.fit(train1_x,train1_y)

pred_rf = model_rf.predict(test1_x)

accuracy_rf=accuracy_score(pred_rf,test1_y)

print(accuracy_rf)

model_ada = AdaBoostClassifier(random_state=100)

model_ada.fit(train1_x,train1_y)

pred_ada =model_ada.predict(test1_x)

accuracy_ada = accuracy_score(pred_ada,test1_y)

print(accuracy_ada)
final = pd.DataFrame({'Decision Tree':[accuracy_dt],

                     'Random Forest':[accuracy_rf],

                     'AdaBoost':[accuracy_ada]})

final
train_x=train.drop("label",axis=1)

train_y=train["label"]

model=RandomForestClassifier(random_state=100,n_estimators=1000)

model.fit(train_x,train_y)

test_predict= model.predict(test)

dt_predict=pd.DataFrame({"Label":test_predict})

dt_predict["ImageId"]=test.index+1

dt_predict[["ImageId","Label"]].to_csv("prediction.csv",index=False)