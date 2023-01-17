%matplotlib inline

import pandas as pd

import numpy as np

from time import time

from fbprophet import Prophet

import logging

import pdb

import random

import os

import matplotlib.pyplot as plt
np.random.seed(7)

Lags =49

N_prophets_trials = 10
train = pd.read_csv("../input/train_1.csv")

train_flattened = pd.melt(train[list(train.columns[-Lags:])+['Page']], id_vars='Page', var_name='date', value_name='Visits')

train_flattened['date'] = train_flattened['date'].astype('datetime64[ns]')

train_flattened['weekend'] = ((train_flattened.date.dt.dayofweek)//5 == 1).astype(float)
test = pd.read_csv("../input/key_1.csv")

test['date'] = test.Page.apply(lambda a: a[-10:])

test['Page'] = test.Page.apply(lambda a: a[:-11])

test['date'] = test['date'].astype('datetime64[ns]')

test['weekend'] = ((test.date.dt.dayofweek) //5 == 1).astype(float)

grouped = train_flattened.groupby(['Page','weekend'])

median_preds = grouped.median().reset_index()

test = test.merge(median_preds, how='left')
train.dropna(axis=0,inplace=True)

fig = plt.figure(figsize=(15,15))

urne = set(range(train.shape[0]))

for k in range(N_prophets_trials):

    i = random.sample(urne,1)[0]

    urne.remove(i)

    ex=train.iloc[i,1:]

    page = train.iloc[i,0]

    df = pd.DataFrame(columns = ['ds','y'])

    df['y'] = ex.values

    df['ds'] = ex.index

    m = Prophet()

    m.fit(df)

    future = m.make_future_dataframe(periods=60)

    preds = m.predict(future)['yhat'][-60:].values

    plt.subplot(5,2,k+1)

    time_serie = list(df['y'].values) + list(preds)

    plt.plot(time_serie)

    plt.plot([550,550],[0,max(time_serie)])

    plt.title(page)

    test.loc[test.Page==page,'Visits'] = preds
test.loc[test.Visits.isnull(), 'Visits']=0

test[['Id','Visits']].to_csv('mad.csv', index=False)