import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.preprocessing import OneHotEncoder

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVR
filepath = '../input/train.csv'

df = pd.read_csv(filepath, sep=',')

df.head()
df = df.replace({'StateHoliday': {'0': 0, 'a': 1, 'b': 2, 'c': 3}})
filepath = '../input/store.csv'

df_store = pd.read_csv(filepath, sep=',')

df_store.head()
df_store = df_store.replace({'StoreType': {'a': 0, 'b': 1, 'c': 2, 'd': 3}})

df_store = df_store.replace({'Assortment': {'a': 0, 'b': 1, 'c': 2}})
df_full = df.merge(df_store, on='Store', how='left')

df_full.head()
filepath = '../input/test.csv'

df_test = pd.read_csv(filepath, sep=',').set_index('Id')

df_test.head()
df_test = df_test.replace({'StateHoliday': {'0': 0, 'a': 1, 'b': 2, 'c': 3}})
df_test.isnull().loc[:,'Open'].value_counts()
df_test.Open.fillna(value=1, inplace=True)

df_test['Open'] = df_test['Open'].astype(int)

df_test['Open'].unique()
df_test_full = df_test.merge(df_store, on='Store', how='left')

df_test_full.head()
df_full['Year'] = df_full['Date'].apply(lambda x: int(x[:4]))

df_full['Month'] = df_full['Date'].apply(lambda x: int(x[5:7]))

df_full['Day'] = df_full['Date'].apply(lambda x: int(x[8:10]))



df_test_full['Year'] = df_test_full['Date'].apply(lambda x: int(x[:4]))

df_test_full['Month'] = df_test_full['Date'].apply(lambda x: int(x[5:7]))

df_test_full['Day'] = df_test_full['Date'].apply(lambda x: int(x[8:10]))
df_full['CompetitionOpen'] = ((df_full['CompetitionOpenSinceYear'] < df_full['Year']) |\

                             ((df_full['CompetitionOpenSinceYear'] == df_full['Year']) &\

                              (df_full['CompetitionOpenSinceMonth'] <= df_full['Month']))).astype(int)



df_test_full['CompetitionOpen'] = ((df_test_full['CompetitionOpenSinceYear'] < df_test_full['Year']) |\

                             ((df_test_full['CompetitionOpenSinceYear'] == df_test_full['Year']) &\

                              (df_test_full['CompetitionOpenSinceMonth'] <= df_test_full['Month']))).astype(int)
df_full['FullDate'] = pd.to_datetime(df_full.Date, format='%Y-%m-%d')

df_test_full['FullDate'] = pd.to_datetime(df_test_full.Date, format='%Y-%m-%d')



import datetime

df_full['WeekNum'] = df_full['FullDate'].apply(func=lambda x: x.isocalendar()[1])

df_test_full['WeekNum'] = df_test_full['FullDate'].apply(func=lambda x: x.isocalendar()[1])
df_full.head().T
columns = ['Store', 'DayOfWeek', 'Open', 'Promo', 'StateHoliday', 'SchoolHoliday', 'StoreType', 'Assortment',

           'CompetitionDistance', 'CompetitionOpenSinceYear', 'Promo2', 'Year', 'Month', 'Day', 'Promo2Now',

           'WeekNum']

X = np.array(df_full.loc[:,columns]).astype(int)

y = np.array(df_full.loc[:,'Sales']).astype(int)

X_test = np.array(df_test_full.loc[:,columns]).astype(int)
cls = RandomForestRegressor(n_estimators=20, criterion='mse').fit(X, y)

score = cls.predict(X_test)
features = {}

for i in range(len(columns)):

    features[columns[i]] = cls.feature_importances_[i]

    

import operator

sorted_features = sorted(features.items(), key=operator.itemgetter(1), reverse=True)

sorted_features
df_out = pd.DataFrame(score, columns=['Sales'])

df_out = df_out.reset_index().rename(index=str, columns={'index': 'Id'})

df_out.loc[:,'Id'] += 1

df_out.head()