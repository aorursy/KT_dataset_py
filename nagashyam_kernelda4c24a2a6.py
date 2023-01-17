# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
train_data = pd.read_csv("../input/train.csv")
test_data = pd.read_csv("../input/test.csv")

# Any results you write to the current directory are saved as output.
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix 
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
train_data.head()
# Splitting the train data into test and train 
train, test = train_test_split(train_data, 
                               test_size = 0.3,
                              random_state = 100)
print(train.shape)
print(test.shape)
train.head()

train_y = train['label'] 
test_y = test['label']

train_x = train.drop('label', axis = 1)
test_x = test.drop('label', axis = 1)
train_x.shape

# Applying the Decision Tree model to find the accuracy
model_dt = DecisionTreeClassifier(max_depth = 5, random_state=100)
model_dt.fit(train_x,train_y)

test_pred = model_dt.predict(test_x)
print(len(test_pred))
test_pred[:5]

df_pred = pd.DataFrame({'actual' : test_y,
                       'predicted' : test_pred})
df_pred.head()

df_pred['pred_status'] = df_pred['actual'] == df_pred['predicted']
df_pred.head()

df_pred[df_pred['pred_status']==True].shape[0] / df_pred.shape[0] *100
# Applying the Random Forest model to find the accuracy
model_rf = RandomForestClassifier(random_state=100, n_estimators = 1000)
model_rf.fit(train_x,train_y)

test_pred = model_rf.predict(test_x)
print(len(test_pred))
test_pred[:5]

df_pred = pd.DataFrame({'actual' : test_y,
                       'predicted' : test_pred})
df_pred.head()

df_pred['pred_status'] = df_pred['actual'] == df_pred['predicted']
df_pred.head()

df_pred[df_pred['pred_status']==True].shape[0] / df_pred.shape[0] *100
# Applying the Ada Boost model to find the accuracy
model_ada = AdaBoostClassifier(random_state=100)
model_ada.fit(train_x,train_y)

test_pred = model_ada.predict(test_x)
print(len(test_pred))
test_pred[:5]

df_pred = pd.DataFrame({'actual' : test_y,
                       'predicted' : test_pred})
df_pred.head()

df_pred['pred_status'] = df_pred['actual'] == df_pred['predicted']
df_pred.head()

df_pred[df_pred['pred_status']==True].shape[0] / df_pred.shape[0] *100
# Applying the best model for the test data to predict it and creating the csv file
train_y = train_data['label'] 
train_x = train_data.drop('label', axis = 1)

model_rf.fit(train_x,train_y)
test_pred = model_rf.predict(test_data)

df_test_pred = pd.DataFrame(test_pred, columns = ['label'])
df_test_pred['ImageId'] = test_data.index+1
df_test_pred[['ImageId','label']].to_csv('submission.csv',index = False)
df_test_pred['ImageId'].shape