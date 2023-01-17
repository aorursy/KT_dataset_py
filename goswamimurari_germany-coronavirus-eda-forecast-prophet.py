# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.dates as mdates

import plotly.express as px

import plotly.graph_objs as go

from plotly.subplots import make_subplots

from fbprophet.plot import plot_plotly, add_changepoints_to_plot

import plotly.offline as py

from datetime import date, timedelta

from statsmodels.tsa.arima_model import ARIMA

from sklearn.cluster import KMeans

from fbprophet import Prophet



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
covid19_df=pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv")

covid19_df = covid19_df.rename(columns={"ObservationDate": "date","Country/Region": "country", "Province/State": "state", "Confirmed":"confirm", "Deaths": "death","Recovered":"recover"})

covid19_df.head()

covid19_df.shape
covid19_df.isnull().sum()
daily_df = covid19_df.sort_values(['date', 'country', 'state'])

latest_data = covid19_df[covid19_df.date == daily_df.date.max()]

latest_data.sample(10)
columns_list = ["state", "country", "date", "confirm", "death", "recover"]

latest_data = covid19_df[columns_list]

latest_data.sample(10)
latest_data_groupby_country = latest_data.groupby("country")[["confirm", "death", "recover"]].sum().reset_index()

latest_data_groupby_country.sample(5)
fig = px.bar(latest_data_groupby_country, 

             y="confirm", x="country", color='country', 

             hover_data = ['confirm', 'death', 'recover'],

             log_y=True, template='ggplot2')

annotations = []

annotations.append(dict(xref='paper', yref='paper', x=0.0, y=1.05,

                              xanchor='left', yanchor='bottom',

                              text='Confirmed bar plot on Country',

                              font=dict(family='Arial',

                                        size=30,

                                        color='rgb(37,37,37)'),

                              showarrow=False))

fig.update_layout(annotations=annotations)

fig.show()
covid19_df['country'].unique()
fig = px.bar(covid19_df.loc[covid19_df['country'] == 'Mainland China'], x='date', y='confirm', 

             hover_data=['state', 'confirm', 'recover'], color='state', template='ggplot2')

annotations = []

annotations.append(dict(xref='paper', yref='paper', x=0.0, y=1.05,

                              xanchor='left', yanchor='bottom',

                              text='Confirmed bar plot for Mainland China over time',

                              font=dict(family='Arial',

                                        size=30,

                                        color='rgb(37,37,37)'),

                              showarrow=False))

fig.update_layout(annotations=annotations)

fig.show()
fig = px.bar(latest_data_groupby_country, 

             y="recover", x="country", color='country', 

             hover_data = ['confirm', 'death', 'recover'],

             log_y=True, template='ggplot2')

annotations = []

annotations.append(dict(xref='paper', yref='paper', x=0.0, y=1.05,

                              xanchor='left', yanchor='bottom',

                              text='Recovered bar plot on Country',

                              font=dict(family='Arial',

                                        size=30,

                                        color='rgb(37,37,37)'),

                              showarrow=False))

fig.update_layout(annotations=annotations)

fig.show()
fig = px.bar(latest_data_groupby_country, 

             y="death", x="country", color='country', 

             hover_data = ['confirm', 'death', 'recover'],

             log_y=True, template='ggplot2')

annotations = []

annotations.append(dict(xref='paper', yref='paper', x=0.0, y=1.05,

                              xanchor='left', yanchor='bottom',

                              text='Death bar plot on Country',

                              font=dict(family='Arial',

                                        size=30,

                                        color='rgb(37,37,37)'),

                              showarrow=False))

fig.update_layout(annotations=annotations)

fig.show()
covid19_df.head()
de_data = covid19_df[covid19_df['country'] == 'Germany']

de_data.tail(5)
prophet_de_confirmed=de_data.iloc[: , [4,5 ]]

prophet_de_confirmed.head()

prophet_de_confirmed.columns = ['ds','y']

prophet_de_confirmed.head()
model_de_confirmed = Prophet()

model_de_confirmed.fit(prophet_de_confirmed)

future_de_confirmed = model_de_confirmed.make_future_dataframe(periods=365)

future_de_confirmed.sample(10)
forecast_de_confirmed=model_de_confirmed.predict(future_de_confirmed)

forecast_de_confirmed.sample(5)
figure_de_confirmed = model_de_confirmed.plot(forecast_de_confirmed)
figure_de_confirmed_2 = model_de_confirmed.plot_components(forecast_de_confirmed)
py.init_notebook_mode()



figure_de_confirmed_2 = plot_plotly(model_de_confirmed, forecast_de_confirmed)  # This returns a plotly Figure

py.iplot(figure_de_confirmed_2)
prophet_de_recover=covid19_df.iloc[: , [4,7 ]]

prophet_de_recover.head()

prophet_de_recover.columns = ['ds','y']

prophet_de_recover.tail()
model_de_recover=Prophet()

model_de_recover.fit(prophet_de_recover)

future_de_recover=model_de_recover.make_future_dataframe(periods=365)

forecast_de_recover=model_de_recover.predict(prophet_de_recover)

forecast_de_recover.sample(5)
figure_de_recover_1 = plot_plotly(model_de_recover, forecast_de_recover)

py.iplot(figure_de_recover_1) 



figure_de_recover_2 = model_de_recover.plot(forecast_de_recover,xlabel='Date',ylabel='Recovery Count')
figure_de_recover_3=model_de_recover.plot_components(forecast_de_recover)
prophet_de_death = covid19_df.iloc[:, [4, 6]]

prophet_de_death.columns=['ds', 'y']

prophet_de_death.tail(5)
model_de_death=Prophet()

model_de_death.fit(prophet_de_death)

future_de_death=model_de_death.make_future_dataframe(periods=365)

forecast_de_death=model_de_death.predict(future_de_death)

forecast_de_death.sample(5)
figure_de_death_1 = plot_plotly(model_de_death, forecast_dth)

py.iplot(figure_de_death_1) 

    

figure_de_death_2 = model_de_death.plot(forecast_de_death,xlabel='Date',ylabel='Death Count')
figure_de_recover_3 = model_de_death.plot_components(forecast_de_death)