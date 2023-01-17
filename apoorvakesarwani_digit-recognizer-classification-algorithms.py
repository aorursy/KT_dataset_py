# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
data=pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/test.csv')
sample=pd.read_csv('../input/sample_submission.csv')

# Any results you write to the current directory are saved as output.
# Splitting data into train and validate
from sklearn.model_selection import train_test_split
train, validate=train_test_split(data, test_size=0.2, random_state=100)
train_x=train.drop('label',axis=1)
train_y=train['label']
validate_x=validate.drop('label', axis=1)
validate_y=validate['label']
#print(train_x.shape, train_y.shape, validate_x.shape, validate_y.shape)
from sklearn.metrics import accuracy_score
## Decision Tree model
## Identified the optimum depth as 19 with most accuracy
from sklearn.tree import DecisionTreeClassifier

model_dt=DecisionTreeClassifier(max_depth=19, random_state=100)
model_dt.fit(train_x, train_y)
validate_pred_dt=model_dt.predict(validate_x)
accuracy_dt=accuracy_score(validate_y, validate_pred_dt)
#print(accuracy_dt)
## Random Forest
## Identified n_estimators as 500 with most accuracy
from sklearn.ensemble import RandomForestClassifier
model_rf=RandomForestClassifier(random_state=100,n_estimators=500)
model_rf.fit(train_x,train_y)
validate_pred_rf=model_rf.predict(validate_x)
accuracy_rf=accuracy_score(validate_y, validate_pred_rf)
#print(accuracy_rf)
## AdaBoost
from sklearn.ensemble import AdaBoostClassifier
model_ab=AdaBoostClassifier(random_state=100)
model_ab.fit(train_x,train_y)
validate_pred_ab=model_ab.predict(validate_x)
accuracy_ab=accuracy_score(validate_y, validate_pred_ab)
#print(accuracy_ab)
## KNN
## KNN with 3 neighbors gives the most accuracy
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
model_knn=KNeighborsClassifier(n_neighbors=3)
model_knn.fit(train_x, train_y)
validate_pred_knn=model_knn.predict(validate_x)
accuracy_knn=accuracy_score(validate_y, validate_pred_knn)
#print(accuracy_knn)
# Using KNN for final prediction as it is providing the most accuracy.
# Could have used Random Forest as it is giving almost same accuracy with less response time.
data_x=data.drop('label',axis=1)
data_y=data['label']
model_knn.fit(data_x,data_y)
test_pred=model_knn.predict(test)

pred_df=pd.DataFrame({'ImageId':test.index.values+1,'Label':test_pred})
pred_df.to_csv('submission_3.csv', index=False)