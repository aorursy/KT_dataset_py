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
train = pd.read_csv('../input/sf-crime/train.csv', parse_dates=['Dates'])
test = pd.read_csv('../input/sf-crime/test.csv', parse_dates=['Dates'], index_col='Id')

train['Day'] = train['Dates'].dt.day
train['DayOfWeek'] = train['Dates'].dt.weekday
train['Month'] = train['Dates'].dt.month
train['Year'] = train['Dates'].dt.year
train['Hour'] = train['Dates'].dt.hour
train['Minute'] = train['Dates'].dt.minute

test['Day'] = test['Dates'].dt.day
test['DayOfWeek'] = test['Dates'].dt.weekday
test['Month'] = test['Dates'].dt.month
test['Year'] = test['Dates'].dt.year
test['Hour'] = test['Dates'].dt.hour
test['Minute'] = test['Dates'].dt.minute
from sklearn.preprocessing import LabelEncoder
le1 = LabelEncoder()
train['PdDistrict'] = le1.fit_transform(train['PdDistrict'])
test['PdDistrict'] = le1.transform(test['PdDistrict'])

le2 = LabelEncoder()
y = le2.fit_transform(train['Category'])
le3 = LabelEncoder()
le3.fit(list(train['Address']) + list(test['Address']))
train['Address'] = le3.transform(train['Address'])
test['Address'] = le3.transform(test['Address'])

train.drop(['Dates', 'Descript', 'Resolution', 'Category'], 1, inplace=True)
test.drop(['Dates'], 1, inplace=True)
from lightgbm import LGBMClassifier
model = LGBMClassifier(n_estimators = 1)
train.head()
test.head()
model.fit(train, y, categorical_feature=['PdDistrict', 'DayOfWeek'])
preds = model.predict_proba(test)
submission = pd.read_csv("../input/sf-crime/sampleSubmission.csv")
submission.iloc[:, 1:] = preds
