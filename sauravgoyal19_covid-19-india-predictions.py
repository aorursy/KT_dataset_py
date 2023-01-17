import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import plotly.express as px

import plotly.graph_objects as go



import warnings

warnings.filterwarnings('ignore')



%matplotlib inline
df=pd.read_csv('../input/covid-19-indiacsv/covid_19_india.csv')

df.head()
df['Total Cases'] = df['ConfirmedIndianNational'] + df['ConfirmedForeignNational']
df['Total Active'] = df['Confirmed'] - (df['Cured'] + df['Deaths'])

total_active = df['Total Active'].sum()

print("Total number of active cases across India : ",total_active)
dbd_India = pd.read_csv('../input/covid-19-india/Covid cases in India.csv',parse_dates=True)
dbd_India.head()
dbd_India['Total Active'] = dbd_India['Total Confirmed cases'] - (dbd_India['Cured/Discharged/Migrated'] + dbd_India['Deaths'])

total_active = dbd_India['Total Active'].sum()

print("Total number of active cases across India : ",total_active)
Tot_cases = df.groupby('State/UnionTerritory')['Total Active'].sum().sort_values(ascending=False).to_frame()

Tot_cases.style.background_gradient(cmap="Blues")
Tot_cases = dbd_India.groupby('Name of State / UT')['Total Confirmed cases'].sum().sort_values(ascending=False).to_frame()

Tot_cases.style.background_gradient(cmap="Reds")
dbd_India.head()
f, ax = plt.subplots(figsize=(12,12))

data = dbd_India[['Name of State / UT','Total Confirmed cases','Cured/Discharged/Migrated','Deaths']]

data.sort_values('Total Confirmed cases',ascending=False, inplace=True)



sns.set_color_codes('pastel')

sns.barplot(x="Total Confirmed cases", y="Name of State / UT", data=data, label="Total", color="r")



sns.set_color_codes("muted")

sns.barplot(x="Cured/Discharged/Migrated", y="Name of State / UT", data=data, label="Cured", color="g")



ax.legend(ncol=2, loc="lower right", frameon=True)

ax.set(ylabel="States/UT",xlabel="Cured",facecolor="white")

sns.despine(left=True, bottom=True)
fig=go.Figure()

fig.add_trace(go.Scatter(x=df['Date'], y=dbd_India['Total Active'], mode='lines+markers',name ='Total Active'))

fig.update_layout(title_text="Coronvirus Cases in India") #plot_bgcolor='rgb(230,230,230)') 

fig.show()              

from fbprophet import Prophet
confirmed = df.groupby('Date').sum()['Confirmed'].reset_index()

deaths = df.groupby('Date').sum()['Deaths'].reset_index()

recovered = df.groupby('Date').sum()['Cured'].reset_index()

confirmed.head()
confirmed.sort_values("Date",axis=0, ascending=True, inplace=False)
confirmed.columns = ['ds', 'y']

confirmed['ds'] = pd.to_datetime(confirmed['ds'])
m = Prophet(interval_width=0.95)

m.fit(confirmed)

future = m.make_future_dataframe(periods=7)

future.head()
future.sort_values("ds",axis=0, ascending=True, inplace=False)
forecast = m.predict(future)

forecast[['ds','yhat', 'yhat_lower','yhat_upper']].tail()
confirmed_forecast_plot = m.plot(forecast)

axes = confirmed_forecast_plot.get_axes()

axes[0].set_xlabel('Date')

axes[0].set_ylabel('Cases')
confirmed_forecast_plot = m.plot_components(forecast)
deaths.columns = ['ds', 'y']

deaths['ds'] = pd.to_datetime(deaths['ds'])
m = Prophet()

m.fit(deaths)

future = m.make_future_dataframe(periods=7)

future.head()
future.sort_values("ds",axis=0, ascending=True, inplace=False)
forecast = m.predict(future)

forecast[['ds','yhat', 'yhat_lower','yhat_upper']].tail()
deaths_forecast_plot = m.plot(forecast)

axes = deaths_forecast_plot.get_axes()

axes[0].set_xlabel('Date')

axes[0].set_ylabel('Cases')
deaths_forecast_plot = m.plot_components(forecast)