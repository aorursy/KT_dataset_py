from sklearn.linear_model import HuberRegressor 

from sklearn.preprocessing import PolynomialFeatures

from sklearn.pipeline import make_pipeline

from matplotlib import pyplot as plt

from datetime import datetime, timedelta

import json

import requests

import pandas as pd

import numpy as np

import plotly.graph_objects as go

import tqdm

import warnings



warnings.simplefilter('ignore')
INPUT_PREFIX = '../input/novel-corona-virus-2019-dataset/time_series_covid_19'



df_confirmed = pd.read_csv(

    INPUT_PREFIX + '_confirmed.csv').drop(['Province/State', 'Lat', 'Long'], axis=1).groupby(by='Country/Region').sum().reset_index()

df_recovered = pd.read_csv(INPUT_PREFIX + '_recovered.csv').drop(['Province/State', 'Lat', 'Long'], axis=1).groupby(by='Country/Region').sum().reset_index()

df_deaths = pd.read_csv(INPUT_PREFIX + '_deaths.csv').drop(['Province/State', 'Lat', 'Long'], axis=1).groupby(by='Country/Region').sum().reset_index()
dates = list(df_confirmed.columns)[1::]
df_merged = pd.DataFrame(columns=['country', 'date', 'confirmed', 'recovered', 'deaths'])



for r, row in tqdm.tqdm(df_confirmed.iterrows(), total=df_confirmed.shape[0]):

    for date in dates:

        df_merged = df_merged.append(

            {

                'country': row['Country/Region'],

                'date_raw': date,

                'confirmed': df_confirmed[df_confirmed['Country/Region']==row['Country/Region']][date].values[0],

                'recovered': df_recovered[df_recovered['Country/Region']==row['Country/Region']][date].values[0],

                'deaths': df_deaths[df_deaths['Country/Region']==row['Country/Region']][date].values[0]

            },

            ignore_index=True

        )



df_merged['active'] = df_merged['confirmed'] - df_merged['recovered'] - df_merged['deaths'] 
df_merged['date'] = df_merged['date_raw'].map(lambda d: datetime.strptime(d, '%m/%d/%y'))
df_merged = df_merged[df_merged['country'].isin(['Italy', 'China', 'Spain', 'Portugal', 'France', 'US', 'Japan', 'Brazil'])]
df_merged[df_merged['country']=='Brazil']
k = 100



df_filtered = df_merged[df_merged['confirmed']>=k]



days_from_k_cases = []



for _, row in df_filtered.iterrows():

    days_from_k_cases.append(

        (row['date'] - df_filtered[df_filtered['country'] == row['country']]['date'].min()).days + 1

    )

    

df_filtered['days_from_k_case'] = days_from_k_cases
X_raw, y_raw = [], []



for country, df_country in df_filtered.groupby(by='country'):

    X_raw += list(df_country['days_from_k_case'].values)

    y_raw += list(df_country['active'].values)

    

X = np.array(X_raw).reshape(-1, 1)

y = np.array(y_raw)
model = make_pipeline(PolynomialFeatures(2), HuberRegressor())

model.fit(X, np.log(y))

X_pred = np.arange(0, 100).reshape(-1, 1)

y_pred_log = model.predict(X_pred)

y_pred = np.exp(y_pred_log)

X_pred = X_pred.reshape(1, -1)[0]
fig = go.Figure(

    layout=go.Layout(

        paper_bgcolor='rgba(0,0,0,0)',

        plot_bgcolor='rgba(0,0,0,0)'

    )

)



for country, df_country in df_filtered.groupby(by='country'):

    fig.add_trace(

        go.Scatter(

            x=df_country['days_from_k_case'],

            y=df_country['active'],

            mode='lines',

            name=country

        )

    )



fig.add_trace(

    go.Scatter(

        x=X_pred,

        y=y_pred,

        mode='markers',

        name='Prediction'

    )

)
y_brazil = df_filtered[(df_filtered['country']=='Brazil')]['active']

brazil_model_diff = y_brazil ** (1/y_pred[0:len(y_brazil)])

brazil_model_mean_diff = np.mean(brazil_model_diff)

print('The difference between the model and the real data is %.2f %% in Brazil'%((brazil_model_mean_diff-1)*100))
y_italy = df_filtered[(df_filtered['country']=='Italy')]['active']

italy_model_diff = y_italy ** (1/y_pred[0:len(y_italy)])

italy_model_mean_diff = np.mean(italy_model_diff)

print('The difference between the model and the real data is %.2f %% in Italy'%((italy_model_mean_diff-1)*100))
italy_brazil_diff = italy_model_mean_diff/brazil_model_mean_diff

print('The difference between the model and the real data is %.2f %% higher in Italy than is Brazil'%((italy_brazil_diff-1)*100))
fig = go.Figure(

    layout=go.Layout(

        paper_bgcolor='rgba(0,0,0,0)',

        plot_bgcolor='rgba(0,0,0,0)'

    )

)



fig.add_trace(

    go.Scatter(

        x=X_pred,

        y=y_pred*brazil_model_mean_diff,

        mode='markers',

        name='Brazil Prediction (active cases)',

        marker={

            'color':'black',

            'size': 2

        }

    )

)



fig.add_trace(

    go.Scatter(

        x=X_pred,

        y=y_pred*0.1*brazil_model_mean_diff,

        mode='markers',

        name='Brazil Prediction (active cases with UTI need)',

        marker={

            'color':'orange',

            'size': 4

        }

    )

)



fig.add_trace(

    go.Scatter(

        x=df_filtered[df_filtered['country']=='Italy']['days_from_k_case'],

        y=df_filtered[df_filtered['country']=='Italy']['active'],

        mode='lines',

        name='Italy',

        line={

            'color':'red',

            'width':5

            

        }

    )

)



fig.add_trace(

    go.Scatter(

        x=df_filtered[df_filtered['country']=='Brazil']['days_from_k_case'],

        y=df_filtered[df_filtered['country']=='Brazil']['active'],

        mode='lines',

        name='Brazil',

        line={

            'color':'black',

            'width':5

            

        }

    )

)



fig.add_trace(

    go.Scatter(

        name="Max capacity of UTI in public health system (SUS)",

        x = [0, 100],

        y = [27400, 27400],

    )

)



fig.add_trace(

    go.Scatter(

        name="Max capacity of UTI in public health system (SUS) + private sector",

        x = [0, 100],

        y = [55100, 55100],

    )

)
public_health_limit_reach = min([i+1 for i,v in enumerate(y_pred) if v >= 27400])

df_filtered[(df_filtered['country']=='Brazil')]['date'].min() + timedelta(days=public_health_limit_reach)