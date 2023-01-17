# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
 
# Any results you write to the current directory are saved as output.
train_df.head()
train_df.isnull().sum()
from sklearn.model_selection import train_test_split

from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score
train,test = train_test_split(train_df, test_size=.25, random_state=75)
train_y = train['label']
test_y = test['label']

train_x = train.drop('label',axis=1)
test_x = test.drop('label',axis=1)
ada = AdaBoostClassifier()
ada.fit(train_x,train_y)

ada_pred = ada.predict(test_x)

ada_accu = accuracy_score(test_y,ada_pred)
ada_accu
rf = RandomForestClassifier()
rf.fit(train_x,train_y)
rf_pred = rf.predict(test_x)

rf_accuracy = accuracy_score(test_y,rf_pred)
rf_accuracy
gr = GradientBoostingClassifier()
gr.fit(train_x,train_y)
gr_pred = gr.predict(test_x)

gr_accuracy = accuracy_score(test_y,gr_pred)
gr_accuracy
test_df.head()
gr = GradientBoostingClassifier()
gr.fit(train_x,train_y)
gr_pred = gr.predict(test_df)
gr_pred[:5]
index = test_df.index + 1
df = pd.DataFrame({'ImageId':index, 'Label':gr_pred})
df
df.to_csv("prediction.csv", index = False)
