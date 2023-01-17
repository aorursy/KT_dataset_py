import numpy as np 

import pandas as pd 

import os



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

temperature_submission = pd.read_csv('/kaggle/input/killer-shrimp-invasion/temperature_submission.csv')

test = pd.read_csv('/kaggle/input/killer-shrimp-invasion/test.csv')

train = pd.read_csv('/kaggle/input/killer-shrimp-invasion/train.csv')
temperature_submission.head(5)
test.head(5)
train.head(5)
import seaborn as sns
ax = sns.countplot(train['Presence'])
train.corr()
#ax = sns.lineplot(train['Salinity_today'], train['Presence'])
#ax = sns.lineplot(train['Temperature_today'], train['Presence'])
#ax = sns.lineplot(train['Substrate'], train['Presence'])
#ax = sns.lineplot(train['Depth'], train['Presence'])
#ax = sns.lineplot(train['Exposure'], train['Presence'])
print('Count of missing values in train data = ', train.isnull().sum(axis=1).sum())
print('Count of missing values in test data = ', test.isnull().sum(axis=1).sum())
train = train.fillna(method='ffill')

test = test.fillna(method='ffill')
print('Count of missing values in train data = ', train.isnull().sum(axis=1).sum())
print('Count of missing values in test data = ', test.isnull().sum(axis=1).sum())
from sklearn.ensemble import ExtraTreesClassifier

clf = ExtraTreesClassifier(n_estimators=100, random_state=0).fit(train.drop(['pointid', 'Presence'], axis=1), train['Presence'])
predictions = clf.predict(test.drop(['pointid'], axis=1))
temperature_submission['Presence'] = predictions

temperature_submission.head(5)
temperature_submission.to_csv('temperature_submission.csv', index=False)