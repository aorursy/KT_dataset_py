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
train=pd.read_csv("../input/train.csv")
test=pd.read_csv("../input/test.csv")
sample_submission=pd.read_csv("../input/sample_submission.csv")
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split


from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
train.head()
test.head()
print(train.shape,test.shape)
a=train.iloc[:,1:]
b=train.iloc[:,0]
a.head()

b.head()

from sklearn.model_selection import train_test_split
train1,validate = train_test_split(train,test_size=0.3,random_state=100)

train1.shape
validate.shape
from sklearn.tree import DecisionTreeClassifier
model_dt=DecisionTreeClassifier(random_state=100,max_depth=5)
train_x=train1.drop('label',axis=1)
test_x=validate.drop('label',axis=1)
train_y=train1['label']
test_y=validate['label']
model_dt.fit(train_x,train_y)
pred_dec = model_dt.predict(test_x)

accuracy_dec = accuracy_score(pred_dec,test_y)
print(accuracy_dec)

class_report_dec = classification_report(pred_dec,test_y)
print(class_report_dec)
##random forest
model_random = RandomForestClassifier(random_state = 100, n_estimators = 300)
model_random.fit(train_x,train_y)
pred_random = model_random.predict(test_x)

accuracy_random = accuracy_score(pred_random,test_y)
print(accuracy_random)

class_report_random = classification_report(pred_random,test_y)
print(class_report_random)
##adaboost
model_ada = AdaBoostClassifier()
model_ada.fit(train_x,train_y)
pred_ada = model_ada.predict(test_x)

accuracy_ada = accuracy_score(pred_ada, test_y)
print(accuracy_ada)
class_report_ada = classification_report(pred_ada,test_y)
print(class_report_ada)
df=pd.DataFrame(columns=['DECISION TREE','RANDOM FOREST','ADA BOOST'],index=['ACCURACY'])
df
df['DECISION TREE']['ACCURACY']=accuracy_dec
df['RANDOM FOREST']['ACCURACY']=accuracy_random
df['ADA BOOST']['ACCURACY']=accuracy_ada
df
###best accuracy comes from random forest using random forest for prediction
model_random = RandomForestClassifier(random_state=100,n_estimators=300)
model_random.fit(train_x,train_y)
pred = model_random.predict(test)

df = pd.DataFrame({'Label':pred})
df['ImageId'] = test.index+1
df[['ImageId','Label']].to_csv('Submission_File.csv',index=False)
df.head()

