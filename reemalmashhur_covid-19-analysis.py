# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd



import plotly.express as px

import matplotlib.pyplot as plt

import seaborn as sns



# hide warnings

import warnings

warnings.filterwarnings('ignore')
# from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

# import plotly as py

# import plotly.graph_objs as go



# init_notebook_mode(connected=True) #do not miss this line



# data = [go.Bar(

#         x=["Monday", "Tuesday"],

#         y=[55,100]  )]

# fig = go.Figure(data=data)



# py.offline.iplot(fig)
ts19confirmed = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv')

ts19recover = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv')

ts19deaths = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv')
# confirmed = pd.melt(ts19confirmed, id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'], var_name='Date', value_name='Confirmed')

# recovered = pd.melt(ts19recover, id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'], var_name='Date', value_name='Recovered')

# deaths = pd.melt(ts19deaths, id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'], var_name='Date', value_name='Deaths')



# result = confirmed

# result['Deaths'] = deaths['Deaths'].values

# result['Recovered'] = recovered['Recovered'].values



# new_data = result

# new_data['Date'] = pd.to_datetime(new_data['Date'])

# new_data = new_data.reset_index(drop=True)

# new_data['Active'] = new_data['Confirmed'] - (new_data['Deaths'] + new_data['Recovered'])

# data = new_data
data = pd.read_csv('/kaggle/input/corona-report/covid19_clean_complete.csv', parse_dates=['Date'])

data['Active'] = data['Confirmed'] - (data['Deaths'] + data['Recovered'])

data = data

without_china = data[data['Country/Region'] != 'China']
last_max_data = data.groupby(['Date'])['Confirmed', 'Deaths', 'Recovered', 'Active'].sum()

last_max_data = last_max_data.reset_index()

last_max_data = last_max_data[last_max_data['Date'] == max(last_max_data['Date'])]

last_max_data = last_max_data.reset_index(drop=True)

last_max_data['Deaths %'] = round(100 * last_max_data['Deaths'] / last_max_data['Confirmed'], 2)

last_max_data['Recovered %'] = round(100 * last_max_data['Recovered'] / last_max_data['Confirmed'], 2)

last_max_data['Active %'] = round(100 * last_max_data['Active'] / last_max_data['Confirmed'], 2)

last_max_data.style.background_gradient(cmap='Pastel1')
wc_last_max_data = without_china.groupby(['Date'])['Confirmed', 'Deaths', 'Recovered', 'Active'].sum()

wc_last_max_data = wc_last_max_data.reset_index()

wc_last_max_data = wc_last_max_data[wc_last_max_data['Date'] == max(wc_last_max_data['Date'])]

wc_last_max_data = wc_last_max_data.reset_index(drop=True)

wc_last_max_data['Deaths %'] = round(100 * wc_last_max_data['Deaths'] / wc_last_max_data['Confirmed'], 2)

wc_last_max_data['Recovered %'] = round(100 * wc_last_max_data['Recovered'] / wc_last_max_data['Confirmed'], 2)

wc_last_max_data['Active %'] = round(100 * wc_last_max_data['Active'] / wc_last_max_data['Confirmed'], 2)

wc_last_max_data.style.background_gradient(cmap='Pastel1')
# color pallette

cnf = '#67000d' # confirmed - dark brown

dth = '#ff2e63' # death - red

rec = '#21bf73' # recovered - cyan

act = '#636efa' # active case - yellow
pi_data = last_max_data.melt(id_vars="Date", value_vars=['Active', 'Deaths', 'Recovered'], var_name='Case', value_name='Count')

fig = px.pie(pi_data, values='Count', names='Case', color_discrete_sequence=[act, rec, dth])

fig.show()
pi_data = wc_last_max_data.melt(id_vars="Date", value_vars=['Active', 'Deaths', 'Recovered'], var_name='Case', value_name='Count')

# df = px.pi_data.tips()

fig = px.pie(pi_data, values='Count', names='Case', color_discrete_sequence=[act, rec, dth])

fig.show()
area_data = data.groupby(['Date'])['Deaths', 'Recovered', 'Active'].sum().reset_index()

area_data = area_data.melt(id_vars="Date", value_vars=['Deaths', 'Recovered', 'Active'], var_name='Case', value_name='Count')

fig = px.area(area_data, x="Date", y="Count", color='Case',

             title='Cases over time', color_discrete_sequence = [dth, rec, act])

fig.show()
wc_area_data = without_china.groupby(['Date'])['Deaths', 'Recovered', 'Active'].sum().reset_index()

wc_area_data = wc_area_data.melt(id_vars="Date", value_vars=['Deaths', 'Recovered', 'Active'], var_name='Case', value_name='Count')

fig = px.area(wc_area_data, x="Date", y="Count", color='Case',

             title='Outside China Cases over time', color_discrete_sequence = [dth, rec, act])

fig.show()