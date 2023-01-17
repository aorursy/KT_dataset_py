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
train_data = pd.read_csv("../input/train.csv")
test_data = pd.read_csv("../input/test.csv")
sample_data = pd.read_csv("../input/sample_submission.csv")
print(train_data.shape)
print(test_data.shape)
train_data.head()
from sklearn.model_selection import train_test_split
train1, validate = train_test_split(train_data,test_size=0.3,random_state=100)
train_x = train1.drop('label',axis=1)
train_y = train1['label']

test_x = validate.drop('label',axis=1)
test_y = validate['label']
print(train_x.shape[0])
print(train_y.shape[0])
print(test_x.shape[0])
print(test_y.shape[0])
###Decision Tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix

model_dt = DecisionTreeClassifier(random_state=100,max_depth=11)
model_dt.fit(train_x,train_y)

pred_dt = model_dt.predict(test_x)

print(accuracy_score(test_y,pred_dt))
print(classification_report(pred_dt, test_y))
### Random Forest
from sklearn.ensemble import RandomForestClassifier
model_rf = RandomForestClassifier(random_state = 100,
                                 n_estimators=490)
model_rf.fit(train_x,train_y)
pred_rf = model_rf.predict(test_x)

print(accuracy_score(test_y,pred_rf))
print(classification_report(test_y,pred_rf))
### AdaBoost
from sklearn.ensemble import AdaBoostClassifier
model_ab = AdaBoostClassifier(random_state = 100,
                                 n_estimators=80)
model_ab.fit(train_x,train_y)
pred_ab = model_ab.predict(test_x)

print(accuracy_score(test_y,pred_ab))
print(classification_report(test_y,pred_ab))
### KNN
from sklearn.neighbors import KNeighborsClassifier

model_knn = KNeighborsClassifier(n_neighbors = 3,
                                weights = 'distance')
model_knn.fit(train_x,train_y)
pred_knn = model_knn.predict(test_x)

print('Accuracy : %.4f' % accuracy_score(test_y,pred_knn))
### Best accuracy by KNN
model = KNeighborsClassifier(n_neighbors = 3,
                            weights = 'distance')
trainx = train_data.drop('label',axis=1)
trainy = train_data['label']
model.fit(trainx,trainy)
pred = model.predict(test_data)

pred = pd.DataFrame({'ImageId':range(1,28001),'Label':pred})
print(pred.head())
pred.to_csv('Submission_File.csv',index=False)