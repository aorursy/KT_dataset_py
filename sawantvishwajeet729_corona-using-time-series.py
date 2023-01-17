import os

import pandas as pd

import numpy as np
df=pd.read_csv("../input/covid19-global-forecasting-week-2/train.csv")
df.head()
df['Province_State']=df['Province_State'].fillna('_')



df['country_state'] = df['Country_Region']+ df['Province_State']
df.drop(['Id'],axis=1, inplace=True)

countries= df['country_state'].unique()

d = dict(tuple(df.groupby('country_state')))

d['Afghanistan_'].head()
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
f=pd.DataFrame()

g = pd.DataFrame()

col_rem = ['Province_State', 'Country_Region', 'Fatalities', 'country_state']

for k in d.keys():

    data = d[k]

    data.drop(col_rem, axis=1, inplace=True)

    data = data.set_index('Date')

    fit = Holt(data, damped=False).fit(smoothing_level=0.8, smoothing_slope=0.6)

    fcast = fit.forecast(30)

    val = fit.fittedvalues

    f[k] = fcast

    g[k] = val[57:70]
test = pd.read_csv('../input/covid19-global-forecasting-week-2/test.csv')



test['Province_State']=test['Province_State'].fillna('_')



test['country_state'] = test['Country_Region']+ test['Province_State']



test['country_state'].nunique()
f.reset_index(inplace=True)

g.reset_index(inplace=True)



f['index']=f['index'].astype('str')
f['index'].unique()
g['Date'].unique()
country = test['country_state'].unique()

test['ConfirmedCases'] = 0

dates = g['Date']

pp=[]

for i in dates:

    for k in country:

        u = g.loc[g['Date'] == i][k].values

        p = test[(test['Date']==i) & (test['country_state'] ==k)].index[0]

        test.iloc[p,5]=u
country = test['country_state'].unique()

dates = f['index']

pp=[]

for i in dates:

    for k in country:

        u = f.loc[f['index'] == i][k].values

        p = test[(test['Date']==i) & (test['country_state'] ==k)].index[0]

        test.iloc[p,5]=u
d = dict(tuple(df.groupby('country_state')))

d['Afghanistan_'].head()
f=pd.DataFrame()

g = pd.DataFrame()

col_rem = ['Province_State', 'Country_Region', 'country_state','ConfirmedCases']

for k in d.keys():

    data = d[k]

    data.drop(col_rem, axis=1, inplace=True)

    data = data.set_index('Date')

    fit = Holt(data, damped=False).fit(smoothing_level=0.8, smoothing_slope=0.6)

    fcast = fit.forecast(30)

    val = fit.fittedvalues

    f[k] = fcast

    g[k] = val[57:70]
f.reset_index(inplace=True)

g.reset_index(inplace=True)



f['index']=f['index'].astype('str')
country = test['country_state'].unique()

test['Fatalities'] = 0

dates = g['Date']

pp=[]

for i in dates:

    for k in country:

        u = g.loc[g['Date'] == i][k].values

        p = test[(test['Date']==i) & (test['country_state'] ==k)].index[0]

        test.iloc[p,6]=u
country = test['country_state'].unique()

dates = f['index']

pp=[]

for i in dates:

    for k in country:

        u = f.loc[f['index'] == i][k].values

        p = test[(test['Date']==i) & (test['country_state'] ==k)].index[0]

        test.iloc[p,6]=u
test
test.set_index('ForecastId', inplace=True)
test.drop(['Province_State','Country_Region','Date','country_state'], axis=1, inplace=True)
test = test.round(0)
test.to_csv('submission.csv', index=False)