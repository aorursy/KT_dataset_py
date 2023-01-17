# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
sample_submission = pd.read_csv('../input/sample_submission.csv')
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report
# Any results you write to the current directory are saved as output.
train1,validate = train_test_split(train, test_size=0.3,random_state=100)
model_dt = RandomForestClassifier(n_estimators=300)
train_x = train1.drop('label', axis=1)
train_y = train1['label']
test_x = validate.drop('label', axis=1)
test_y = validate['label']

model_randfor = RandomForestClassifier(random_state=100,n_estimators = 400)
model_randfor.fit(train_x, train_y)

test_predict = model_randfor.predict(test_x)
accuracy_randfor = accuracy_score(test_y,test_predict)
accuracy_randfor
test = pd.read_csv("../input/test.csv")
test_predict = model_randfor.predict(test)

dataframe_predict = pd.DataFrame(test_predict,columns=['Label'])
dataframe_predict['ImageId'] = test.index + 1
dataframe_predict.head(5)

prediction = dataframe_predict[['ImageId', 'Label']].to_csv('prediction.csv', index = False)

pd.read_csv('prediction.csv')
