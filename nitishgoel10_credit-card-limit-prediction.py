# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

df = pd.read_csv("../input/CREDIT CARD USERS DATA.csv")

df.head()



# Any results you write to the current directory are saved as output.
null_columns = df.columns[df.isin(['#NULL!']).any()].tolist()

for feature in null_columns:

    print(feature, len(df[df[feature]=='#NULL!']))

    # df.isnull().values.any()

df = df.drop(['custid','lntollten', 'lntollmon', 'lnequipmon','lnequipten', 'lncardmon','lnwiremon', 'lnwireten','lncardten' ],axis=1)

df.head()
df['total_spend'] = df['cardspent'] + df['card2spent']

df.drop(['cardspent', 'card2spent'], axis=1, inplace=True)
for feature in ['townsize', 'cardten']:

    mode_value = df[df[feature]!='#NULL!'][feature].mode().iloc[0]

    df.loc[df[feature]=='#NULL!', feature]=mode_value

for features in ['lncreddebt', 'lnothdebt', 'commutetime','longten','lnlongten']:

    mean_value = df[df[features]!='#NULL!'][features].astype(float).mean()

    df.loc[df[features]=='#NULL!', features]=mean_value

# for feature in null_columns:

#     print(feature, len(df[df[feature]=='#NULL!'])
co_list= df.corr().abs().unstack().sort_values()['total_spend']

# print(co_list)



pd.set_option('display.max_columns', None)  # or 1000

pd.set_option('display.max_rows', None)  # or 1000

pd.set_option('display.max_colwidth', -1)

print(co_list)
df.drop(['pets', 'commutepublic', 'reside', 'cartype', 'carbought', 'pets_small', 'commutecarpool', 'pets_cats', 'card2type', 'commutewalk',

        'hometype', 'commutecar', 'pets_freshfish', 'cardfee', 'churn', 'card2benefit', 'union', 'commute', 'cars', 'pets_saltfish', 'commuterail',

        'reason', 'active'], axis=1, inplace=True)
df2 = pd.get_dummies(df, columns=['region', 'birthmonth', 'edcat','jobcat', 'spousedcat', 'address', 

                            'carown', 'carcatvalue', 'commutecat',

                           'polview','card', 'cardtype','cardbenefit','card2', 'bfast', 'internet'

                            ])
from sklearn import preprocessing

from sklearn.model_selection import train_test_split



# df.dtypes

x = df2.loc[:, df2.columns != 'total_spend']

y = df2['total_spend'].values

x = preprocessing.normalize(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=3)
from sklearn.linear_model import LinearRegression, Ridge, Lasso

from sklearn import metrics

from sklearn.preprocessing import PolynomialFeatures

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.svm import SVR



regressor = LinearRegression()

regressor.fit(x_train,y_train)

linear_predicted_Values = regressor.predict(x_test)

print(y_test)

print(linear_predicted_Values)

print(regressor.score(x_train, y_train))

print(regressor.score(x_test, y_test))
co_list= df['total_spend'].sort_values()

# print(co_list)



pd.set_option('display.max_columns', None)  # or 1000

pd.set_option('display.max_rows', None)  # or 1000

pd.set_option('display.max_colwidth', -1)

print(co_list)