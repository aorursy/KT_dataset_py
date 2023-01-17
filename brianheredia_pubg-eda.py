# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/train.csv')
print(df.head())
print(list(df))
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(df.corr(), 
            annot=True, 
            linewidths=.7, 
            ax=ax,
            fmt='.1f',
            cmap=sns.cm.rocket_r)
df = df.drop(columns = ['groupId', 'maxPlace', 'numGroups', 'roadKills', 'teamKills'])
print(list(df))
high_corr = ['walkDistance', 'killPlace', 'boosts', 'weaponsAcquired']
for item in high_corr:
    print('The average for {} was {:.2f}'.format(item, df[item].mean()))
    print('The standard deviation for {} was {:.2f}'.format(item, df[item].std()))
    print('The minimum for {} was {}'.format(item, df[item].min()))
    print('The maximum for {} was {}'.format(item, df[item].max()))
    print('The correlation between win placement and {} is {:.2f}'.format(item, df[item].corr(df['winPlacePerc'])))
    print('________________________')
test_df = pd.read_csv('../input/test.csv')

y_train = df['winPlacePerc']
x_train = df[['walkDistance','weaponsAcquired', 'boosts', 'killPlace']]
x_test = test_df[['walkDistance','weaponsAcquired', 'boosts', 'killPlace']]

print(x_train.head())
scaler = MinMaxScaler()
x_train.loc[:,['walkDistance','weaponsAcquired', 'boosts', 'killPlace']] = scaler.fit_transform(x_train.loc[:,['walkDistance','weaponsAcquired', 'boosts', 'killPlace']])
x_test.loc[:,['walkDistance','weaponsAcquired', 'boosts', 'killPlace']] = scaler.fit_transform(x_test.loc[:,['walkDistance','weaponsAcquired', 'boosts', 'killPlace']])
print(x_train.head())

print(x_train.shape, y_train.shape, x_test.shape)
clf = LinearRegression()
clf = clf.fit(x_train, y_train)
fit_score = clf.score(x_train, y_train)
yhat_test = clf.predict(x_test)
print('R2 for train data: {:.4f}'.format(fit_score))
submission = pd.DataFrame({'Id':test_df['Id'], 
                           'winPlacePerc':yhat_test},
                            columns = ['Id','winPlacePerc'])
submission.to_csv('LR_submission.csv', index=False)