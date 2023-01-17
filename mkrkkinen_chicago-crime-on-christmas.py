import pandas as pd

import numpy as np

import scipy.stats as stats

import pymc3 as pm

import matplotlib.pyplot as plt

!pip install arviz

import arviz as az
az.style.use('arviz-darkgrid')
import bq_helper

from bq_helper import BigQueryHelper

# https://www.kaggle.com/sohier/introduction-to-the-bq-helper-package

chicago_crime = bq_helper.BigQueryHelper(active_project="bigquery-public-data",

                                   dataset_name="chicago_crime")
bq_assistant = BigQueryHelper("bigquery-public-data", "chicago_crime")

bq_assistant.list_tables()
bq_assistant.head("crime", num_rows=3)
bq_assistant.table_schema("crime")
weather = pd.read_csv("../input/historical-hourly-weather-data/temperature.csv", 

                      usecols=["datetime", "Chicago"], error_bad_lines=False)
weather.head()
weather.info()
weather[weather.isna().any(axis = 1)].head(10)
weather1= weather.dropna()
weather1['datetime'].min()
weather1['datetime'].max()
violent = ['ASSAULT','BATTERY','CRIM SEXUAL ASSAULT', 'ROBBERY',

            'HOMICIDE', 'KIDNAPPING']
query1 = """SELECT

    primary_type,

    date

FROM

    `bigquery-public-data.chicago_crime.crime`

WHERE

    arrest = True 

    AND (date > '2012-10-01') AND (date < '2017-11-30')

    AND primary_type IN ('ASSAULT','BATTERY','CRIM SEXUAL ASSAULT', 'ROBBERY',

            'HOMICIDE', 'KIDNAPPING')

"""



response1 = chicago_crime.query_to_pandas_safe(query1)

response1.head(10)
response1[response1.isna().any(axis=1)]
response1['datetime'] = pd.to_datetime(response1['date']).dt.tz_localize(None)

response2 = response1.set_index(['datetime'])
response2.head(3)
response2 = response2.resample('D').count()

response2.head()
weather1.head(3)
weather1['datetime'] = pd.to_datetime(weather1['datetime'])

weather2 = weather1.set_index(['datetime'])

weather2 = weather2.resample('D').mean()
weather2.head(3)
df = weather2.join(response2, on='datetime')



df.columns=['Ktemp', 'incidents', 'incidents2']
df.head()
df.info()
df=df.dropna()
plt.scatter(df['Ktemp'], df['incidents'])
with pm.Model() as model1: 

    alpha = pm.Normal('alpha',20,10)

    beta = pm.Normal('beta',0.2,0.1)

    sigma = pm.Uniform('sigma', 0,10)

    mu = pm.Deterministic('mu', alpha + beta* df['Ktemp'])

    pred = pm.Normal('pred', mu,sigma, observed = df['incidents'])

    trace1 = pm.sample(1000,tune = 1000, cores=2)
az.plot_trace(trace1, var_names =['~mu'])
x_points = np.linspace(250,320,100)

plt.scatter(df['Ktemp'], df['incidents'], alpha = 0.5)

plt.plot(x_points, trace1['alpha'].mean() + trace1['beta'].mean() * x_points, color = 'red')

mu_pred = trace1['alpha'] + trace1['beta'] * x_points[:,None]

prediction = stats.norm.rvs(mu_pred, trace1['sigma'])

az.plot_hpd(x_points, prediction.T, credible_interval=0.95)
df2 = df[(df.index.month == 12) & (df.index.day.isin([24,25,26,27,28,29,30,31]))]
df2.head()
with pm.Model() as model2: 

    alpha = pm.Normal('alpha',20,10)

    beta = pm.Normal('beta',0.2,0.1)

    sigma = pm.Uniform('sigma', 0,10)

    mu = pm.Deterministic('mu', alpha + beta* df2['Ktemp'])

    pred = pm.Normal('pred', mu,sigma, observed = df2['incidents'])

    trace2 = pm.sample(1000,tune = 1000, cores=2)
az.plot_trace(trace2, var_names=['~mu'])
x_points = np.linspace(250,290,100)

plt.scatter(df2['Ktemp'], df2['incidents'], alpha = 0.5)

plt.plot(x_points, trace2['alpha'].mean() + trace2['beta'].mean() * x_points, color = 'red')

mu_pred = trace2['alpha'] + trace2['beta'] * x_points[:,None]

prediction = stats.norm.rvs(mu_pred, trace2['sigma'])

az.plot_hpd(x_points, prediction.T, credible_interval = 0.95)
x_points = np.linspace(250,310,100)

plt.scatter(df['Ktemp'], df['incidents'], alpha = 0.5)

plt.scatter(df2['Ktemp'], df2['incidents'], alpha = 1, color = 'red')

plt.plot(x_points, trace1['alpha'].mean() + trace1['beta'].mean() * x_points, color = 'blue')

mu_pred1 = trace1['alpha'] + trace1['beta'] * x_points[:,None]

prediction1 = stats.norm.rvs(mu_pred, trace1['sigma'])

az.plot_hpd(x_points, prediction1.T, color = 'lightblue')

x_points2 = np.linspace(254,285,100)

mu_pred2 = trace2['alpha'] + trace2['beta'] * x_points2[:,None]

prediction2 = stats.norm.rvs(mu_pred2, trace2['sigma'])

plt.plot(x_points2, trace2['alpha'].mean() + trace2['beta'].mean() * x_points, color = 'red')

az.plot_hpd(x_points2, prediction2.T, credible_interval = 0.95, color = 'red')