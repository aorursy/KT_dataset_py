# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from fbprophet import Prophet



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
covid_confirm = pd.read_csv("../input/confirmed.csv")

covid_confirm
covid_recov = pd.read_csv("../input/recovered.csv")

covid_recov
covid_death = pd.read_csv("../input/death.csv")

covid_death
Country = covid_death['Country/Region']

Country
# extracting data only for Canada 

Canada_data = covid_death.loc[Country=='Canada']

Canada_data
#pre-processing the dataset

Canada_deaths = Canada_data.iloc[:, 4:]

Canada_deaths
#Summation of death each day

Daily_Deaths = Canada_deaths.sum()

Daily_Deaths
# count of death each day

Canada_deaths.loc['Total Deaths'] = Daily_Deaths

Canada_deaths
Dates = list(Canada_deaths.columns.values)

Dates_dataf = pd.DataFrame(Dates)

Dates_dataf
total_death = list(Canada_deaths.T['Total Deaths'])

total_death = pd.DataFrame(total_death)

total_death
deaths_per_dates = pd.concat([Dates_dataf, total_death], axis =1)

deaths_per_dates.columns =['ds', 'y']

deaths_per_dates
# forecasting the death for upcoming 10 days

prediction = Prophet(interval_width = 0.90)

prediction.fit(deaths_per_dates)

Future_deathtoll = prediction.make_future_dataframe(periods = 10)

Future_counts = prediction.predict(Future_deathtoll)



Future_counts
Future_data = Future_counts[['ds', 'yhat', 'yhat_lower','yhat_upper']]

Future_data
prediction.plot(Future_data)
# forecasting the death for upcoming 20 days

prediction = Prophet(interval_width = 0.90)

prediction.fit(deaths_per_dates)

Future_deathtoll = prediction.make_future_dataframe(periods = 20)

Future_counts = prediction.predict(Future_deathtoll)



Future_data = Future_counts[['ds', 'yhat', 'yhat_lower','yhat_upper']]
prediction.plot(Future_data)
# forecasting the death for upcoming 30 days

prediction = Prophet(interval_width = 0.90)

prediction.fit(deaths_per_dates)

Future_deathtoll = prediction.make_future_dataframe(periods = 30)

Future_counts = prediction.predict(Future_deathtoll)



Future_data = Future_counts[['ds', 'yhat', 'yhat_lower','yhat_upper']]
prediction.plot(Future_data)
# forecasting the death for next 1 year

prediction = Prophet(interval_width = 0.90)

prediction.fit(deaths_per_dates)

Future_deathtoll = prediction.make_future_dataframe(periods = 365)

Future_counts = prediction.predict(Future_deathtoll)



Future_data = Future_counts[['ds', 'yhat', 'yhat_lower','yhat_upper']]
prediction.plot(Future_data)
from fbprophet.plot import plot_plotly

import plotly.offline as py

py.init_notebook_mode()



# forecasting the death for next 30 days

prediction = Prophet(interval_width = 0.90)

prediction.fit(deaths_per_dates)

Future_deathtoll = prediction.make_future_dataframe(periods = 30)

Future_counts = prediction.predict(Future_deathtoll)



Future_data = Future_counts[['ds', 'yhat', 'yhat_lower','yhat_upper']]



fig = plot_plotly(prediction, Future_data)  # This returns a plotly Figure

py.iplot(fig)