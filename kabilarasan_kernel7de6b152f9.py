# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

data = pd.read_csv("../input/train.csv")



# Any results you write to the current directory are saved as output.
data.head(5)
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score,classification_report

from xgboost import XGBClassifier
train,test = train_test_split(data)

train_x = train.drop('label',axis =1)

train_y = train['label']



test_x = test.drop('label',axis =1)

test_y = test['label']
#AdaBoostClassifier

model_ab = AdaBoostClassifier(n_estimators = 500)

model_ab.fit(train_x,train_y)

predict_ab = model_ab.predict(test_x)

accuracy_score(test_y,predict_ab)
model_rf = RandomForestClassifier(n_estimators = 500,random_state =100)

model_rf.fit(train_x,train_y)

predict_rf = model_rf.predict(test_x)
### xgboost

xgb_model = XGBClassifier()

xgb_model.fit(train_x,train_y)

xgb_result = xgb_model.predict(test_x)
accuracy_score(test_y,xgb_result)
test = pd.read_csv("../input/test.csv")

test_digit_recog = xgb_model.predict(test)



df_pred_digit = pd.DataFrame(test_digit_recog, columns=['Label'])

df_pred_digit['ImageId'] = test.index + 1

df_pred_digit.head(10)