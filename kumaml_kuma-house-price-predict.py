# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import LabelEncoder, MinMaxScaler

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
datapath = '../input/'

df_train = pd.read_csv(datapath+'train.csv')

df_test = pd.read_csv(datapath+'test.csv')

print(df_train.shape)

print(df_test.shape)
df_train.head()
ids = df_test['Id']

train_Y = np.log1p(df_train['SalePrice'])

df_train = df_train.drop(['Id', 'SalePrice'], axis=1)

df_test = df_test.drop(['Id'], axis=1)
sns.regplot(x=df_train['GrLivArea'], y=train_Y)

plt.show()
# df_train['GrLivArea'] = df_train['GrLivArea'].clip(800, 2500)

# sns.regplot(x=df_train['GrLivArea'], y=train_Y)

# plt.show()
IsMatched = (df_train['GrLivArea'] > 800) & (df_train['GrLivArea'] < 2500)

IsMatched.head()
df_train = df_train[IsMatched]

df_train
train_Y = train_Y[IsMatched]
sns.regplot(x=df_train['GrLivArea'], y=train_Y)

plt.show()
sns.regplot(x=df_train['1stFlrSF'], y=train_Y)

plt.show()
IsMatched = (df_train['1stFlrSF'] > 500) & (df_train['1stFlrSF'] < 2250)

IsMatched.head()
df_train = df_train[IsMatched]

train_Y = train_Y[IsMatched]
sns.regplot(x=df_train['1stFlrSF'], y=train_Y)

plt.show()
sns.distplot(df_train['LotArea'])

plt.show()
df_train['LotArea'] = np.log1p(df_train['LotArea'])

df_test['LotArea'] = np.log1p(df_test['LotArea'])

sns.distplot(df_train['LotArea'])

plt.show()
df = pd.concat([df_train, df_test])

df.head()
lblencoder = LabelEncoder()



for c in df.columns:

    df[c] = df[c].fillna(-1)

    if df[c].dtype == 'object':

        df[c] = lblencoder.fit_transform(list(df[c].values))

df.head()
mmscaler = MinMaxScaler()



for c in df.columns:

    df[c] = mmscaler.fit_transform(df[c].values.reshape(0,1))

#     df[c] = mmscaler.fit_transform(df[c].values.reshape(-1,1))

df.head()
trainnum = train_Y.shape[0]

train_X = df[:trainnum]

test_X = df[trainnum:]
from sklearn.linear_model import LinearRegression



model = LinearRegression()

model.fit(train_X, train_Y)

pred = model.predict(test_X)
pred = np.expm1(pred)
submission = pd.DataFrame({'Id':ids, 'SalePrice':pred})

submission.head()
submission.to_csv('submission.csv', index=False)