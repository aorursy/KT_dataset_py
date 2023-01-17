# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # data visualization



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv("../input/train.csv")

test_data = pd.read_csv("../input/test.csv")

sample_data = pd.read_csv("../input/sampleSubmission.csv")
train_data.head()
test_data.head()
train_data.info()
test_data.info()
train_data.duplicated().sum()
train_data.drop_duplicates(inplace=True)
X = train_data.drop(labels=['Category', 'Descript', 'Resolution', 'Address'], axis=1).copy()

y = train_data['Category'].copy()
def split_date(df):

    df['Hour'] = df['Dates'].apply(lambda x: x.split()[1].split(':')[0]).astype(int)

    df['Year'] = df['Dates'].apply(lambda x: x.split()[0].split('-')[0]).astype(int)

    df['Month'] = df['Dates'].apply(lambda x: x.split()[0].split('-')[1]).astype(int)

    df['Day'] = df['Dates'].apply(lambda x: x.split()[0].split('-')[2]).astype(int)

    return df.drop(labels='Dates', axis=1)
X = split_date(X)
X.info()
X.DayOfWeek.unique()
def map_weekday(df):

    week = {'Sunday': 0, 'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 'Friday': 5, 'Saturday': 6}

    df['DayOfWeek'] = df['DayOfWeek'].map(week)

    return df
X = map_weekday(X)
X.head()
X.DayOfWeek[:10000].plot()
def cyclical_features(df):

    df['Sin_DayOfWeek'] = np.sin(df.DayOfWeek*(2.*np.pi/7))

    df['Sin_Hour'] = np.sin(df.Hour*(2.*np.pi/24))

    df['Sin_Month'] = np.sin(df.Month*(2.*np.pi/12))

    df['Sin_Day'] = np.sin(df.Day*(2.*np.pi/31))

    return df.drop(labels=['DayOfWeek', 'Hour', 'Month', 'Day'], axis=1)
X = cyclical_features(X)
X.head()
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
X['PdDistrict'] = encoder.fit_transform(X['PdDistrict'])
X.head()
# from sklearn.preprocessing import OneHotEncoder

# onehot = OneHotEncoder(categories=[sample_data.columns[1:]], sparse=False)

# y = onehot.fit_transform(y.values.reshape(-1, 1))
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from xgboost import XGBClassifier
xgb_clf = XGBClassifier(tree_method='gpu_hist', random_state=42)
X_train.shape, y_train.shape
xgb_clf.fit(X_train, y_train)
from sklearn.metrics import log_loss
y_proba = xgb_clf.predict_proba(X_test)

log_loss(y_test, y_proba)
xgb_clf.fit(X, y)
test_X = test_data.drop(labels=['Id', 'Address'], axis=1)
test_X = split_date(test_X)

test_X = map_weekday(test_X)

test_X = cyclical_features(test_X)

test_X['PdDistrict'] = encoder.transform(test_X['PdDistrict'])
test_X.head()
submission = sample_data.copy()
answer = xgb_clf.predict_proba(test_X)
submission.iloc[:, 1:] = answer
submission.info()
submission.to_csv('submission.csv', sep=',', index=False)