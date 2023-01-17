import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import datetime
df = pd.read_csv('../input/covid19-global-forecasting-week-1/train.csv', parse_dates=['Date'], index_col='Id')

df.rename(columns={'Date': 'date',

                     'Province/State':'state',

                     'Country/Region':'country',

                     'Lat':'lat',

                     'Long': 'long',

                     'ConfirmedCases': 'confirmed',

                     'Fatalities':'deaths',

                    }, inplace=True)

df.head()
df = df.drop(['state'], axis=1).groupby(['country','date']).sum().reset_index()

confirmed = df.pivot_table(index='date', columns='country', values='confirmed')

confirmed.tail()
drop_countries = confirmed.sum()==.0

confirmed.drop(drop_countries.index[drop_countries==True].tolist(), axis=1, inplace=True)

confirmed.tail()
confirmed_corr = confirmed_norm.corr()

confirmed_corr['Brazil'].sort_values(ascending=False).head(10)
first_BR = (confirmed['Brazil']>10).idxmax()

days_since_first = confirmed.index[-1] - first_BR

print(first_BR)
drop_countries_1 = confirmed.loc[first_BR] < 1

confirmed_BR = confirmed.drop(drop_countries_1.index[drop_countries_1==True].tolist(), axis=1)



drop_countries_2 = confirmed_BR.iloc[-1] <= confirmed_BR.iloc[-1].loc['Brazil']

confirmed_BR = confirmed_BR.drop(drop_countries_2.index[drop_countries_2==True].tolist(), axis=1)



confirmed_BR_countries = confirmed_BR.columns.tolist()

confirmed_BR.tail()
aligned = pd.DataFrame()

aligned['Brazil'] = confirmed.loc[first_BR:,'Brazil'].values

aligned.head()
for country in confirmed_BR_countries:

    first = (confirmed[country]>10).idxmax()

    aligned[country] = confirmed.loc[first:first+days_since_first,country].values

    print(first, country)

    

aligned.head()
aligned.drop(['Austria', 'Belgium', 'Netherlands', 'Portugal'], axis=1, inplace=True)

fig,ax = plt.subplots(figsize=(10,10))

aligned.plot(ax=ax, cmap='tab20')
aligned.corr()['Brazil'].sort_values(ascending=False)
aligned.drop(['Germany','Korea, South'], axis=1, inplace=True)
from sklearn.linear_model import LinearRegression



X = aligned.drop('Brazil', axis=1)

y = aligned['Brazil']



lm = LinearRegression()

lm.fit(X,y)

coefs = lm.coef_

print(lm.score(X,y))
sns.barplot(y=X.columns.tolist(),x=coefs)
aligned_new = pd.DataFrame()

days_rest = days_since_first + datetime.timedelta(days=6)



for country in X.columns.tolist():

    first = (confirmed[country]>10).idxmax()

    aligned_new[country] = confirmed.loc[first:first+days_rest,country].values

    

aligned_new.tail()
aligned_new['Brazil_pred'] = lm.predict(aligned_new)
fig, ax = plt.subplots(figsize=(12,12))

aligned_new.plot(cmap='tab20', ax=ax)