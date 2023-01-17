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
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
print(train.shape)
print(test.shape)
train.head()
train.isnull().sum()
train.target.value_counts()
train['target'] = train['target'].replace({"Class_1":1,"Class_2":2,"Class_3":3,"Class_4":4,"Class_5":5,"Class_6":6,"Class_7":7,"Class_8":8,"Class_9":9})
train['target'] = train['target'].astype('category')
numericFeatures = train._get_numeric_data().columns.tolist()
len(numericFeatures)
train.drop('id',axis=1,inplace=True)
train.head()
X = train.drop('target',axis=1)
y = train['target']
X.shape
from sklearn.preprocessing import RobustScaler
robust_scaler = RobustScaler()
X = robust_scaler.fit_transform(X)

from sklearn.ensemble import ExtraTreesClassifier

clf = ExtraTreesClassifier()
clf.fit(X, y)
test.head()
id = test['id']
X_test = test.drop('id',axis=1)
# Create submission
X_test = robust_scaler.transform(X_test)
Y_pred = clf.predict_proba(X_test)

submission = pd.DataFrame({ "id": id})
range_of_classes = range(1, 10)
i = 0
# Create column name based on target values(see sample_submission.csv)
for num in range_of_classes:
    col_name = str("Class_{}".format(num))
    submission[col_name] = Y_pred[:,i]
    i = i + 1
    
submission.to_csv('otto.csv', index=False)
submission.head()
