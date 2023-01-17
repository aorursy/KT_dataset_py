import numpy as np

import pandas as pd

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import missingno as mn

import seaborn as sns

import matplotlib.pyplot as plt

import plotly.express as px
df_cpi = pd.read_csv('../input/us-consumer-price-index/US_CPI.csv')

df_home = pd.read_csv('../input/san-francisco-housing-income-cpi-workers/Home_Price_index.csv')

df_income = pd.read_csv('../input/san-francisco-housing-income-cpi-workers/Income.csv')

df_rent = pd.read_csv('../input/san-francisco-housing-income-cpi-workers/Rent.csv')

df_worker = pd.read_csv('../input/san-francisco-housing-income-cpi-workers/Workers_Index.csv')
display(df_cpi.head())

display(df_home.head())

display(df_income.head())

display(df_rent.head())

display(df_worker.head())
df_cpi['Date'] = pd.to_datetime(df_cpi['Date'], format = '%m/%d/%Y')

df_home['Date'] = pd.to_datetime(df_home['Date'], format = '%Y-%m-%d')

df_income['Date'] = pd.to_datetime(df_income['Date'], format = '%m/%d/%Y')

df_rent['Date'] = pd.to_datetime(df_rent['Date'], format = '%m/%d/%Y')

df_worker['Date'] = pd.to_datetime(df_worker['Date'], format = '%Y-%m-%d')



df_cpi = df_cpi.set_index(df_cpi['Date'])

df_home = df_home.set_index(df_home['Date'])

df_income = df_income.set_index(df_income['Date'])

df_rent = df_rent.set_index(df_rent['Date'])

df_worker = df_worker.set_index(df_worker['Date'])



df_cpi = df_cpi.drop(columns = ['Date'])

df_home = df_home.drop(columns = ['Date'])

df_income = df_income.drop(columns = ['Date'])

df_rent = df_rent.drop(columns = ['Date'])

df_worker = df_worker.drop(columns = ['Date'])
df1 = pd.concat([df_home, df_rent, df_worker], axis = 1)

mn.matrix(df1)
df1 = df1[df1.index >= '1990-01-01']
df1 = df1.fillna(method = 'ffill')

mn.matrix(df1)
fig = px.line(df1, x=df1.index, y=df1.columns)

    

fig.update_layout(

        template='gridon',

        title='Bay Area Housing vs Workers Index',

        xaxis_title='Year',

        yaxis_title = 'Index Values',

        xaxis_showgrid=True,

        yaxis_showgrid=False,

    shapes = [

                    dict(

            type="rect",

            x0='2000-03-01',

            x1='2002-10-01',

            y0=df1['Home Price Index'].min(),

            y1=df1['Rent Index'].max(),

            fillcolor="Red",

            opacity=0.3,

            layer="below",

            line_width=0

                    ),

                dict(

            type="rect",

            x0="2007-12-01",

            y0=df1['Home Price Index'].min(),

            x1="2009-06-01",

            y1=df1['Rent Index'].max(),

            fillcolor="Red",

            opacity=0.3,

            layer="below",

            line_width=0,

            )

        ],

        annotations=[

             dict(text="The Great Recession",x = '2007-12-01', y=df1['Rent Index'].max()),

             dict(text="Dot Com Bubble", x='2000-03-01', y=df1['Rent Index'].max())

         ]

    )

fig.show()
df2 = df1.drop(columns = ['Rent Index'])

fig = px.line(df2, x=df2.index, y=df2.columns)

    

fig.update_layout(

        template='gridon',

        title='Bay Area House Prices vs Workers Index',

        xaxis_title='Year',

        yaxis_title = 'Index Values',

        xaxis_showgrid=True,

        yaxis_showgrid=False,

    shapes = [

                    dict(

            type="rect",

            x0='2000-03-01',

            x1='2002-10-01',

            y0=df2['Home Price Index'].min(),

            y1=df2['Home Price Index'].max(),

            fillcolor="Salmon",

            opacity=0.5,

            layer="below",

            line_width=0

                    ),

                dict(

            type="rect",

            x0="2007-12-01",

            y0=df2['Home Price Index'].min(),

            x1="2009-06-01",

            y1=df2['Home Price Index'].max(),

            fillcolor="Red",

            opacity=0.4,

            layer="below",

            line_width=0,

            )

        ],

        annotations=[

             dict(text="The Great Recession",x = '2007-12-01', y=df2['Home Price Index'].max()),

             dict(text="Dot Com Bubble", x='2000-03-01', y=df2['Home Price Index'].max())

         ]

    )

fig.show()
df_all = pd.concat([df_cpi, df_home, df_income, df_rent, df_worker], axis = 1)
df_all = df_all[df_all.index >= '1990-01-01']

df_all = df_all[df_all.index <= '2018-01-01']
df_all = df_all.fillna(method = 'ffill')

df_all
sns.pairplot(df_all, corner=True, kind='reg', diag_kind='kde', plot_kws=dict(scatter_kws=dict(s=6)))

plt.show()
df_all = df_all.drop(columns = ['US Workers Index', 'San Fran Workers Index'])
date = df_all.index

from sklearn.preprocessing import MinMaxScaler

df_all = df_all.astype('float')

scaler = MinMaxScaler()

df_scaled = pd.DataFrame(scaler.fit_transform(df_all), columns=df_all.columns)

df_scaled = df_scaled.set_index(date)
fig = px.line(df_scaled, x=df_scaled.index, y=df_scaled.columns)

    

fig.update_layout(

        template='gridon',

        title='Bay Area Housing Cost vs CPI vs Income',

        xaxis_title='Year',

        yaxis_title = 'Scaled Values',

        xaxis_showgrid=True,

        yaxis_showgrid=False,

    shapes = [

                    dict(

            type="rect",

            x0='2000-03-01',

            x1='2002-10-01',

            y0=0,

            y1=df_scaled.Income.max(),

            fillcolor="LightSalmon",

            opacity=0.5,

            layer="below",

            line_width=0

                    ),

                dict(

            type="rect",

            x0="2007-12-01",

            y0=0,

            x1="2009-06-01",

            y1=df_scaled.Income.max(),

            fillcolor="Red",

            opacity=0.4,

            layer="below",

            line_width=0,

            )

        ],

        annotations=[

             dict(text="The Great Recession",x = '2007-12-01', y=df_scaled.Income.max()),

             dict(text="Dot Com Bubble", x='2000-03-01', y=df_scaled.Income.max())

         ]

    )

fig.show()
from fbprophet import Prophet

from fbprophet.plot import plot_plotly, plot_components_plotly



model_df = pd.DataFrame(df_home).reset_index()

model_df = model_df.rename(columns={'Date': 'ds', 'Home Price Index': 'y'})



# Initialise the model and make predictions

m = Prophet()

m.fit(model_df)



future = m.make_future_dataframe(periods=24, freq='M')



forecast = m.predict(future)



plot_plotly(m, forecast)
plot_components_plotly(m, forecast)