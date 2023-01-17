import numpy as np 
import pandas as pd 

import os
print(os.listdir("../input"))
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
sample_submission = pd.read_csv('../input/sample_submission.csv')

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier 

df_train, df_validate = train_test_split(train, test_size=0.3,random_state=100)

## Decision Tree Classifier
model_dt = DecisionTreeClassifier(random_state=100)
train_x = df_train.drop('label', axis=1)
train_y = df_train['label']
validate_x = df_validate.drop('label', axis=1)
validate_y = df_validate['label']
model_dt.fit(train_x, train_y)
validate_pred = model_dt.predict(validate_x)
accuracy_score(validate_y, validate_pred)

model_rf = RandomForestClassifier(random_state=100, n_estimators=1000)
train_x = df_train.drop('label', axis=1)
train_y = df_train['label']
validate_x = df_validate.drop('label', axis=1)
validate_y = df_validate['label']
model_rf.fit(train_x, train_y)
validate_pred = model_rf.predict(validate_x)
accuracy_score(validate_y, validate_pred)
model_ab = AdaBoostClassifier(random_state=100, n_estimators=1000)
train_x = df_train.drop('label', axis=1)
train_y = df_train['label']
validate_x = df_validate.drop('label', axis=1)
validate_y = df_validate['label']
model_ab.fit(train_x, train_y)
validate_pred = model_ab.predict(validate_x)
accuracy_score(validate_y, validate_pred)
test_pred = model_rf.predict(test)
df_test_pred = pd.DataFrame(test_pred, columns=['Label'])
df_test_pred['ImageId'] = test.index + 1
df_test_pred[["ImageId","Label"]].to_csv("Submission.csv",index=False)
