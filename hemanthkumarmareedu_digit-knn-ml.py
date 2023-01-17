# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

train=pd.read_csv('../input/train.csv')
#print(train.head())

test=pd.read_csv('../input/test.csv')
#print(test.head())
# Any results you write to the current directory are saved as output.
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report
df_train,df_validate=train_test_split(train,
                                     train_size=0.7,
                                     random_state=100)
print(train.shape)
print(df_train.shape)
print(df_validate.shape)

print(test.shape)

#Using DecisionTree Algorithm
from sklearn.tree import DecisionTreeClassifier

train_x=df_train.drop('label',axis=1)
train_y=df_train['label']

validate_x=df_validate.drop('label',axis=1)
validate_y=df_validate['label']

model_dt=DecisionTreeClassifier(max_depth=5)
model_dt.fit(train_x,train_y)


validate_pred=model_dt.predict(validate_x)

print(accuracy_score(validate_y,validate_pred))
#Using RandomForest Algorithm
from sklearn.ensemble import RandomForestClassifier
train_x=df_train.drop('label',axis=1)
train_y=df_train['label']

validate_x=df_validate.drop('label',axis=1)
validate_y=df_validate['label']

model_rf=RandomForestClassifier(max_depth=5)
model_rf.fit(train_x,train_y)


validate_pred=model_rf.predict(validate_x)

print(accuracy_score(validate_y,validate_pred))
#Using AdaBoost Algorithm
from sklearn.ensemble import AdaBoostClassifier
train_x=df_train.drop('label',axis=1)
train_y=df_train['label']

validate_x=df_validate.drop('label',axis=1)
validate_y=df_validate['label']

model_ab=AdaBoostClassifier(random_state=100)
model_ab.fit(train_x,train_y)


validate_pred=model_ab.predict(validate_x)

print(accuracy_score(validate_y,validate_pred))
#Using KNN Algorithm
from sklearn import neighbors
train_x=df_train.drop('label',axis=1)
train_y=df_train['label']



validate_x=df_validate.drop('label',axis=1)
validate_y=df_validate['label']

model_knn=neighbors.KNeighborsClassifier(n_neighbors=3)
model_knn.fit(train_x,train_y)


validate_pred=model_knn.predict(validate_x)

print(accuracy_score(validate_y,validate_pred))
#Using Naive Bayes Algorithm
from sklearn.naive_bayes import GaussianNB
train_x=df_train.drop('label',axis=1)
train_y=df_train['label']

validate_x=df_validate.drop('label',axis=1)
validate_y=df_validate['label']

model_nb = GaussianNB()
model_nb.fit(train_x,train_y)


validate_pred=model_nb.predict(validate_x)

print(accuracy_score(validate_y,validate_pred))
# Predicting the ouput for test dataset using KNN model
test_pred = model_knn.predict(test)

df_test_pred = pd.DataFrame(test_pred, columns=['Label'])
df_test_pred['ImageId'] = test.index + 1

df_test_pred[['ImageId', 'Label']].to_csv('sample_submission.csv', index=False)
