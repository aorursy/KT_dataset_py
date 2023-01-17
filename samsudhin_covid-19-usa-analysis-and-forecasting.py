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
import warnings

warnings.filterwarnings("ignore")



import numpy as np

import pandas as pd



import plotly.graph_objects as go

import plotly.express as px

import plotly.io as pio

pio.templates.default = "plotly_dark"

from plotly.subplots import make_subplots



from pathlib import Path

data_dir = Path('/kaggle/input/')



import os

print(os.listdir(data_dir))



from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error, mean_absolute_error

from matplotlib import pyplot as plt

%matplotlib inline
state_population = pd.read_csv('../input/us-statewise-population.csv')#https://www.census.gov/data/datasets/time-series/demo/popest/2010s-state-total.html

state_population.columns = ['stateid','state','population']

state_population.drop(['stateid'],axis=1,inplace=True)

state_corona = pd.read_csv('../input/us-cleaned-corona-data.csv', parse_dates=['date'])#https://www.kaggle.com/imdevskp/corona-virus-report.

state_corona.state[state_corona.state == 'Puerto Rico'] = 'Puerto Rico Commonwealth'

for st in list(set(state_corona.state.unique()) - set(state_population.state)):

    state_corona.drop(state_corona[state_corona.state == st]. index, axis=0, inplace=True)

state_corona = state_corona.merge(state_population,on='state')
# Check if the data is updated

print("External Data")

print(f"Earliest Entry: {state_corona['date'].min()}")

print(f"Last Entry:     {state_corona['date'].max()}")

print(f"Total Days:     {state_corona['date'].max() - state_corona['date'].min()}")
group = state_corona.groupby('date')['date', 'confirmed', 'deaths'].sum().reset_index()

fig = px.line(group, x="date", y="confirmed", title="US Confirmed Cases Over Time")

fig.show()
fig = px.line(group, x="date", y="deaths", title="US Deaths Over Time")

fig.show()
cleaned_latest = state_corona[state_corona['date'] == max(state_corona['date'])]

flg = cleaned_latest.groupby('state')['confirmed', 'population'].agg({'confirmed':'sum', 'population':'mean'}).reset_index()



flg['infectionRate'] = round((flg['confirmed']/flg['population'])*100, 5)

temp = flg[flg['confirmed']>100]

temp = temp.sort_values('infectionRate', ascending=False)



fig = px.bar(temp.sort_values(by="infectionRate", ascending=False)[:10][::-1],

             x = 'infectionRate', y = 'state', 

             title='% of infected people by state', text='infectionRate', height=800, orientation='h',

             color_discrete_sequence=['red']

            )

fig.show()
formated_gdf = state_corona.groupby(['date', 'state','Lat','Long'])['confirmed', 'population'].max()

formated_gdf = formated_gdf.reset_index()

formated_gdf['date'] = pd.to_datetime(formated_gdf['date'])

formated_gdf['date'] = formated_gdf['date'].dt.strftime('%m/%d/%Y')

formated_gdf['infectionRate'] = round((formated_gdf['confirmed']/formated_gdf['population'])*100, 8)

fig = px.scatter_geo(formated_gdf,

        lon = 'Long',

        lat = 'Lat',

        text = 'state',

        color="infectionRate", 

        size='infectionRate', 

        hover_name="state", 

        range_color= [0, 0.2],

        animation_frame="date", 

        title='COVID-19: Spread Over Time (Normalized by State Population)', 

        color_continuous_scale="portland"

        )

fig.update_layout(

        geo_scope='usa',

    )

fig.show()
cleaned_latest = state_corona[state_corona['date'] == max(state_corona['date'])]

flg = cleaned_latest.groupby('state')['confirmed', 'deaths', 'recovered', 'active'].sum().reset_index()



flg['mortalityRate'] = round((flg['deaths']/flg['confirmed'])*100, 2)

temp = flg[flg['confirmed']>100]

temp = temp.sort_values('mortalityRate', ascending=False)



fig = px.bar(temp.sort_values(by="mortalityRate", ascending=False)[:10][::-1],

             x = 'mortalityRate', y = 'state', 

             title='Deaths per 100 Confirmed Cases', text='mortalityRate', height=800, orientation='h',

             color_discrete_sequence=['darkred']

            )

fig.show()
formated_gdf = state_corona.groupby(['date', 'state','Lat','Long'])['confirmed', 'deaths'].max()

formated_gdf = formated_gdf.reset_index()

formated_gdf['date'] = pd.to_datetime(formated_gdf['date'])

formated_gdf['date'] = formated_gdf['date'].dt.strftime('%m/%d/%Y')

formated_gdf['mortalityRate'] = round((formated_gdf['deaths']/formated_gdf['confirmed'])*100, 8)

formated_gdf['mortalityRate'] = formated_gdf['mortalityRate'].fillna(0)

fig = px.scatter_geo(formated_gdf,

        lon = 'Long',

        lat = 'Lat',

        text = 'state',

        color="mortalityRate", 

        size='mortalityRate', 

        hover_name="state", 

        range_color= [0, 0.2],

        animation_frame="date", 

        title='COVID-19: Spread Over Time (Normalized by State Population)', 

        color_continuous_scale="portland"

        )

fig.update_layout(

        geo_scope='usa',

    )

fig.show()
icu_beds = pd.read_csv('/kaggle/input/us-beds.csv')#https://www.kff.org/other/state-indicator/beds-by-ownership/?currentTimeframe=0&sortModel=%7B%22colId%22:%22Location%22,%22sort%22:%22asc%22%7D

icu_beds.loc[-1] = ['Puerto Rico', 0.0]

icu_beds.index = icu_beds.index + 1

icu_beds = icu_beds.sort_index() 

state_corona = pd.merge(state_corona, icu_beds, on='state')
latest_grouped = state_corona.groupby(['Lat','Long','state'])['TotalBeds(1000)'].mean().reset_index()

fig = px.bar(latest_grouped.sort_values('TotalBeds(1000)', ascending=False)[:10][::-1], 

             x='TotalBeds(1000)', y='state',

             title='Ratio of ICU Beds per 1000 People', text='TotalBeds(1000)', orientation='h',color_discrete_sequence=['green'] )

fig.show()
fig = px.scatter_geo(latest_grouped,

        lon = 'Long',

        lat = 'Lat',

        text = 'state',

        color="TotalBeds(1000)", 

        size='TotalBeds(1000)', 

        hover_name="state", 

        title='Ratio of ICU beds per 1000 people', 

        color_continuous_scale="portland"

        )

fig.update_layout(

        geo_scope='usa',

    )

fig.show()
from fbprophet import Prophet

all_df = state_corona.groupby('date')['confirmed', 'deaths', 'recovered'].sum().reset_index()

df_prophet = all_df.loc[:,["date", 'confirmed']]

df_prophet.columns = ['ds','y']

m_d = Prophet(

    yearly_seasonality= True,

    weekly_seasonality = True,

    daily_seasonality = True,

    seasonality_mode = 'multiplicative')

m_d.fit(df_prophet)

future_d = m_d.make_future_dataframe(periods=14)

fcst_daily = m_d.predict(future_d)

fcst_daily[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
fig1 = m_d.plot(fcst_daily)
from fbprophet.plot import plot_plotly

import plotly.offline as py

py.init_notebook_mode()



fig = plot_plotly(m_d, fcst_daily)  # This returns a plotly Figure

py.iplot(fig)