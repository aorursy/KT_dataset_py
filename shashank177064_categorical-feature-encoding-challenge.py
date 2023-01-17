"""

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

"""
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import accuracy_score
df_train = pd.read_csv('../input/cat-in-the-dat/train.csv')
df_train.head()
features = df_train.drop(['target'], axis = 1)

target = df_train['target']
print(features.shape,target.shape)
df_feature = pd.DataFrame()

le = LabelEncoder()

for c in features.columns:

    df_feature[c] = le.fit_transform(features[c])

    
#df_feature.head()

print(df_feature.shape)
train_f, test_f, train_y, test_y = train_test_split(df_feature,target,test_size=0.2, random_state=42)

print(train_f.shape, train_y.shape)
clf = RandomForestClassifier(n_estimators = 200, n_jobs = -1, verbose = 2)

clf.fit(train_f,train_y)

target_pre = clf.predict(test_f)

print(accuracy_score(target_pre,test_y))
df_test = pd.read_csv('../input/cat-in-the-dat/test.csv')

df_test.head()

df_submission = pd.DataFrame()

df_submission['id'] = df_test['id']
for c in df_test.columns:

    df_test[c] = le.fit_transform(df_test[c])
prediction = clf.predict(df_test)
df_submission['target'] = prediction
df_submission.head()
df_submission.to_csv("submission.csv", index = False)