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
#train.head()
#test=pd.read_csv("../input/test.csv")
#test.head()
sample_submission=pd.read_csv("../input/train.csv")
#sample_submission.head()
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
train,validate = train_test_split(train,test_size = 0.3, random_state =100)
## Decision Tree 
train1_y = train['label']
train1_x = train.drop('label', axis=1)
test1_y = validate['label']
test1_x = validate.drop('label', axis=1)

model = DecisionTreeClassifier(random_state=100)
model.fit(train1_x,train1_y)

test_pred = model.predict(test1_x)
print(accuracy_score(test1_y, test_pred))
## Random Forest
train1_y = train['label']
train1_x = train.drop('label', axis=1)
test1_y = validate['label']
test1_x = validate.drop('label', axis=1)

model_rf = RandomForestClassifier(random_state=100,n_estimators = 1000)
model_rf.fit(train1_x, train1_y)

test_pred_rf = model_rf.predict(test1_x)
print(accuracy_score(test1_y, test_pred_rf))
## Adaptive Boost
train1_y = train['label']
train1_x = train.drop('label', axis=1)
test1_y = validate['label']
test1_x = validate.drop('label', axis=1)

model = AdaBoostClassifier(random_state=100)
model.fit(train1_x,train1_y)

test_pred_Ab = model.predict(test1_x)
print(accuracy_score(test1_y, test_pred_Ab))
## K Nearest Neighbors
train1_y = train['label']
train1_x = train.drop('label', axis=1)
test1_y = validate['label']
test1_x = validate.drop('label', axis=1)

model = KNeighborsClassifier(n_neighbors=5)
model.fit(train1_x,train1_y)

test_pred_knn = model.predict(test1_x)
print(accuracy_score(test1_y, test_pred_knn))
test_pred = model_rf.predict(test)
df_test_pred = pd.DataFrame(test_pred, columns=['Label'])
df_test_pred['ImageId'] = test.index + 1
df_test_pred[['ImageId','Label']].to_csv('Submission.csv', index=False)
