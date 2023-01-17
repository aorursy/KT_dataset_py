# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import LabelEncoder, MinMaxScaler



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
datapath = '../input/'

df_train = pd.read_csv(datapath+'train.csv')

df_test = pd.read_csv(datapath+'test.csv')

df_train.head()
print(df_train.shape)

print(df_test.shape)
df_train.info()
ids = df_test['PassengerId']

train_Y = df_train['Survived']

df_train = df_train.drop(['PassengerId', 'Survived'], axis=1)

df_test = df_test.drop(['PassengerId'], axis=1)

print(df_train.shape)

print(df_test.shape)
df = pd.concat([df_train, df_test])

df.head()
lblencoder = LabelEncoder()

mmscaler = MinMaxScaler()



for c in df.columns:

    df[c] = df[c].fillna(-1)

    if df[c].dtype == 'object':

        df[c] = lblencoder.fit_transform(list(df[c].values))

    df[c] = mmscaler.fit_transform(df[c].values.reshape(-1,1))

df.head()
trainnum = train_Y.shape[0]

train_X = df[:trainnum]

test_X = df[trainnum:]
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(train_X, train_Y)

pred = model.predict(test_X)
submission = pd.DataFrame({'PassengerId':ids, 'Survived':pred})

submission
submission.to_csv('submission.csv', index=False)