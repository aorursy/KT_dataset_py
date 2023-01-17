# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

train =  pd.read_csv("../input/train.csv")
print(train.shape)

test_data = pd.read_csv("../input/test.csv")

# Any results you write to the current directory are saved as output.
train.head()
# Import Some Library

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split


from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
# Here target Value is Label
# Now split the data into train and test

train,test = train_test_split(train, test_size = .33, random_state = 100)
print(train.head())
print(train.shape)
test.shape
#
train_y = train['label']
test_y = test['label']

train_x = train.drop('label',axis = 1)
test_x = test.drop('label', axis=1)
print(train_x.shape[0])
print(train_y.shape[0])
print(test_x.shape[0])
print(test_y.shape[0])
# Decision Tree
model_dec = DecisionTreeClassifier()
model_dec.fit(train_x,train_y)

pred_dec = model_dec.predict(test_x)

accuracy_dec = accuracy_score(pred_dec,test_y)
print(accuracy_dec)

class_report_dec = classification_report(pred_dec,test_y)
print(class_report_dec)
model_random = RandomForestClassifier(random_state = 100, n_estimators=1000)
model_random.fit(train_x,train_y)

pred_random = model_random.predict(test_x)

accuracy_random = accuracy_score(pred_random,test_y)
print(accuracy_random)

class_report_random = classification_report(pred_random,test_y)
print(class_report_random)
model_ada = AdaBoostClassifier()

model_ada.fit(train_x,train_y)

pred_ada = model_ada.predict(test_x)

accuracy_ada = accuracy_score(pred_ada, test_y)
print(accuracy_ada)

class_report_ada = classification_report(pred_ada,test_y)
print(class_report_ada)
#model_knn = KNeighborsClassifier(n_neighbors=21)
model_knn.fit(train_x,train_y)

pred_knn = model_knn.predict(test_x)

accuracy_knn = accuracy_score(pred_knn, test_y)
print(accuracy_knn)
train_x = train.drop('label', axis=1)
train_y  =train['label']
rf = RandomForestClassifier(random_state=100, n_estimators=1000)
rf.fit(train_x,train_y)
pred = rf.predict(test_data)

df = pd.DataFrame({'Label':pred})
df['ImageId'] = test_data.index+1
df[['ImageId','Label']].to_csv('Submission_File.csv',index=False)
df.head()