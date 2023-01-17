# Installs
!pip install pycountry_convert 
!pip install folium
!pip install calmap
!wget https://raw.githubusercontent.com/tarunk04/COVID-19-CaseStudy-and-Predictions/master/models/model_deaths.h5
!wget https://raw.githubusercontent.com/tarunk04/COVID-19-CaseStudy-and-Predictions/master/models/model_confirmed.h5
!wget https://raw.githubusercontent.com/tarunk04/COVID-19-CaseStudy-and-Predictions/master/models/model_usa_c.h5


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker 
import seaborn as sns
import pycountry_convert as pc
import folium
from datetime import datetime, timedelta,date
from scipy.interpolate import make_interp_spline, BSpline
import plotly.express as px
import json, requests
import calmap

import plotly.graph_objects as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot, plot_mpl
import plotly.offline as py
init_notebook_mode(connected=True)
plt.rcParams.update({'font.size': 14})

%matplotlib inline

# Retriving Dataset
confirmed_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
deaths_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')
recovered_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')

data_df = pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/web-data/data/cases_country.csv")

confirmed_df.head()
# Drop date columns if they are mostly NaN kiểm tra các cột ko có dữ liệu
na_columns = (confirmed_df.isna().sum() / confirmed_df.shape[0]) > 0.99
na_columns = na_columns[na_columns]

confirmed_df = confirmed_df.drop(na_columns.index, axis=1)
deaths_df = deaths_df.drop(na_columns.index, axis=1)
recovered_df = recovered_df.drop(na_columns.index, axis=1)
confirmed_df.head()
## Tidy up the data
confirmed_df = confirmed_df.melt(id_vars=['Country/Region', 'Province/State', 'Lat', 'Long'], var_name='date', value_name='confirmed')
deaths_df = deaths_df.melt(id_vars=['Country/Region', 'Province/State', 'Lat', 'Long'], var_name='date', value_name='deaths')
recovered_df = recovered_df.melt(id_vars=['Country/Region', 'Province/State', 'Lat', 'Long'], var_name='date', value_name='recovered')
confirmed_df.head()
recovered_df.head()
confirmed_df['date'] = pd.to_datetime(confirmed_df['date'])
deaths_df['date'] = pd.to_datetime(deaths_df['date'])
recovered_df['date'] = pd.to_datetime(recovered_df['date'])
confirmed_df.head()
#all_df = confirmed_df.merge(recovered_df).merge(deaths_df)
#all_df = all_df.rename(columns={'Country/Region': 'Country', 'date': 'Date', 'confirmed': "Confirmed", "recovered": "Recovered", "deaths": "Deaths"})
#all_df.head()
all_df = confirmed_df.merge(recovered_df).merge(deaths_df)
all_df = all_df.rename(columns={'Country/Region': 'Country', 'date': 'Date', 'confirmed': "Confirmed", "recovered": "Recovered", "deaths": "Deaths"})
# Check null values
all_df.isnull().sum()
all_df.head()
world_df = all_df.groupby(['Date']).agg({'Confirmed': ['sum'], 'Recovered': ['sum'], 'Deaths': ['sum']}).reset_index()
world_df.columns = world_df.columns.get_level_values(0)

def add_rates(df):
    df['Confirmed Change'] = df['Confirmed'].diff().shift(-1)
 
    df['Mortality Rate'] = df['Deaths'] / df['Confirmed']
    df['Recovery Rate'] = df['Recovered'] / df['Confirmed']
    df['Growth Rate'] = df['Confirmed Change'] / df['Confirmed']
    df['Growth Rate Change'] = df['Growth Rate'].diff().shift(-1)
    df['Growth Rate Accel'] = df['Growth Rate Change'] / df['Growth Rate']
    return df

world_df = add_rates(world_df)
# Take a view of Worldwide Cases
def plot_aggregate_metrics(df, fig=None):
    if fig is None:
        fig = go.Figure()
    fig.update_layout(template='plotly_dark')
    fig.add_trace(go.Scatter(x=df['Date'], 
                             y=df['Confirmed'],
                             mode='lines+markers',
                             name='Confirmed',
                             line=dict(color='Yellow', width=2)
                            ))
    fig.add_trace(go.Scatter(x=df['Date'], 
                             y=df['Deaths'],
                             mode='lines+markers',
                             name='Deaths',
                             line=dict(color='Red', width=2)
                            ))
    fig.add_trace(go.Scatter(x=df['Date'], 
                             y=df['Recovered'],
                             mode='lines+markers',
                             name='Recovered',
                             line=dict(color='Green', width=2)
                            ))
    return fig
plot_aggregate_metrics(world_df).show()
# Take a view of Worldwide Rates
def plot_diff_metrics(df, fig=None):
    if fig is None:
        fig = go.Figure()

    fig.update_layout(template='plotly_dark')
    fig.add_trace(go.Scatter(x=df['Date'], 
                             y=df['Mortality Rate'],
                             mode='lines+markers',
                             name='Mortality rate',
                             line=dict(color='red', width=2)))

    fig.add_trace(go.Scatter(x=df['Date'], 
                             y=df['Recovery Rate'],
                             mode='lines+markers',
                             name='Recovery rate',
                             line=dict(color='Green', width=2)))

    fig.add_trace(go.Scatter(x=df['Date'], 
                             y=df['Growth Rate'],
                             mode='lines+markers',
                             name='Confirmed rate',
                             line=dict(color='Yellow', width=2)))
    fig.update_layout(yaxis=dict(tickformat=".1%"))
    
    return fig
plot_diff_metrics(world_df).show()
# Daily Percent Change in Growth Rate
# Useful for tracking whether the growth rate is increasing. Any positive percentage indicates exponential growth.

fig = go.Figure()
fig.update_layout(template='plotly_dark')

tmp_df = world_df.copy()
tmp_df = tmp_df[tmp_df['Growth Rate Accel'] < 10]

fig.add_trace(go.Scatter(x=tmp_df['Date'], 
                         y=tmp_df['Growth Rate Accel'],
                         mode='lines+markers',
                         name='Growth Acceleration',
                         line=dict(color='Green', width=3)))
fig.update_layout(yaxis=dict(tickformat=".1%"))

fig.show()
data_df.head()
data_df = data_df.rename(columns={'Country_Region': 'Country', 'Last_Update': 'Date', 'Lat': 'Lat', 'Long_': 'Long'})
data_df.head()
data_wd = pd.DataFrame(data_df.groupby(['Country', 'Lat', 'Long', 'Date'])['Confirmed', 'Recovered', 'Deaths'].sum()).reset_index()
data_wd
# Check null values
data_wd.isnull().sum()
# top 20 countries by confirmed cases
top20_cf = data_wd.groupby('Country').max().sort_values(by='Confirmed', ascending=False)[:20]
top20_cf.reset_index()
# top 20 countries by death cases
topo20_death = data_wd.groupby('Country').max().sort_values(by='Deaths', ascending=False)[:20]
topo20_death.reset_index()
# color pallette
cnf, dth, rec, act = '#393e46', '#ff2e63', '#21bf73', '#fe9801' 
from plotly.subplots import make_subplots
# confirmed - deaths
fig_c = px.bar(data_wd.sort_values('Confirmed').tail(20), x="Confirmed", y="Country", 
               text='Confirmed', orientation='h', color_discrete_sequence = [act])
fig_d = px.bar(data_wd.sort_values('Deaths').tail(20), x="Deaths", y="Country", 
               text='Deaths', orientation='h', color_discrete_sequence = [dth])


fig = make_subplots(rows=1, cols=2, shared_xaxes=False, horizontal_spacing=0.1, vertical_spacing=5.0,
                    subplot_titles=('Top 20 countries with confirmed cases', 'Top 20 countries with Death cases'))
fig.add_trace(fig_c['data'][0], row=1, col=1)
fig.add_trace(fig_d['data'][0], row=1, col=2)

data_select_agg = all_df.groupby(['Country', 'Date']).sum().reset_index()
def plot_time_variation_countries(df, countries, case_type='Confirmed', size=3, is_log=False):
    f, ax = plt.subplots(1,1, figsize=(4*size,3*size))
    for country in countries:
        df_ = df[(df['Country']==country) & (df['Date'] > '2020-02-01')] 
        g = sns.lineplot(x="Date", y=case_type, data=df_,  label=country)  
        ax.text(max(df_['Date']), max(df_[case_type]), str(country))
    plt.xlabel('Date')
    plt.ylabel(f'Total  {case_type} cases')
    plt.title(f'Total {case_type} cases')
    plt.xticks(rotation=90)
    if(is_log):
        ax.set(yscale="log")
    ax.grid(color='black', linestyle='dotted', linewidth=0.75)
    plt.show()  
countries = [ 'US', 'Spain', 'Italy', 'France', 'Germany', 'United Kingdom', 'China', 'Turkey', 'Iran', 'Russia', 'Belgium', 'Brazil']
plot_time_variation_countries(data_select_agg, countries, size=4)
countries = [ 'US', 'Spain', 'Italy', 'France', 'Germany', 'United Kingdom', 'China', 'Turkey', 'Iran', 'Russia', 'Belgium', 'Brazil']
plot_time_variation_countries(data_select_agg, countries,case_type = 'Deaths', size=4)
data_select_agg = all_df.groupby(['Country', 'Date']).sum().reset_index()
def plot_time_variation_countries_asean(df, countries, case_type='Confirmed', size=3, is_log=False):
    f, ax = plt.subplots(1,1, figsize=(4*size,3*size))
    for country in countries:
        df_ = df[(df['Country']==country) & (df['Date'] > '2020-02-01')] 
        g = sns.lineplot(x="Date", y=case_type, data=df_,  label=country)  
        ax.text(max(df_['Date']), max(df_[case_type]), str(country))
    plt.xlabel('Date')
    plt.ylabel(f'Total  {case_type} cases')
    plt.title(f'1. Total {case_type} cases in ASEAN \n 1. Số Ca dương tính Covid-19 các nước trong ASEAN ')
    plt.xticks(rotation=90)
    if(is_log):
        ax.set(yscale="log")
    ax.grid(color='black', linestyle='dotted', linewidth=0.75)
    plt.show()  
##countries = ['Vietnam','Thailand','Indonesia','Malaysia','Singapore','Philippines','Cambodia','Lao','Myanmar','Brunei']
countries = ['Vietnam', 'Thailand', 'Indonesia', 'Malaysia', 'Singapore', 'Philippines', 'Cambodia','Brunei']
plot_time_variation_countries_asean(data_select_agg, countries, size=4)
def plot_time_variation_countries_asean(df, countries, case_type='Confirmed', size=3, is_log=False):
    f, ax = plt.subplots(1,1, figsize=(4*size,3*size))
    for country in countries:
        df_ = df[(df['Country']==country) & (df['Date'] > '2020-02-01')] 
        g = sns.lineplot(x="Date", y=case_type, data=df_,  label=country)  
        ax.text(max(df_['Date']), max(df_[case_type]), str(country))
    plt.xlabel('Date')
    plt.ylabel(f'Total  {case_type} cases')
    plt.title(f'2. Total {case_type} cases in ASEAN \n 2. Số ca tử vong có dương tính Covid-19 các nước trong ASEAN ')
    plt.xticks(rotation=90)
    if(is_log):
        ax.set(yscale="log")
    ax.grid(color='black', linestyle='dotted', linewidth=0.75)
    plt.show() 
countries = ['Vietnam', 'Thailand', 'Indonesia', 'Malaysia', 'Singapore', 'Philippines', 'Cambodia','Brunei']
plot_time_variation_countries_asean(data_select_agg, countries,case_type = 'Deaths', size=4)
def plot_time_variation_countries_asean(df, countries, case_type='Confirmed', size=3, is_log=False):
    f, ax = plt.subplots(1,1, figsize=(4*size,3*size))
    for country in countries:
        df_ = df[(df['Country']==country) & (df['Date'] > '2020-02-01')] 
        g = sns.lineplot(x="Date", y=case_type, data=df_,  label=country)  
        ax.text(max(df_['Date']), max(df_[case_type]), str(country))
    plt.xlabel('Date')
    plt.ylabel(f'Total  {case_type} cases')
    plt.title(f'3. Total {case_type} cases in ASEAN \n 3. Số ca phục hồi có dương tính Covid-19 các nước trong ASEAN ')
    plt.xticks(rotation=90)
    if(is_log):
        ax.set(yscale="log")
    ax.grid(color='black', linestyle='dotted', linewidth=0.75)
    plt.show() 
countries = ['Vietnam', 'Thailand', 'Indonesia', 'Malaysia', 'Singapore', 'Philippines', 'Cambodia', 'Brunei']
plot_time_variation_countries_asean(data_select_agg, countries,case_type = 'Recovered', size=4)
#world_cases
world_details = pd.pivot_table(all_df, values=['Confirmed','Deaths','Recovered'], index='Country', aggfunc='max')
world_details['Recovery Rate'] = round(world_details['Recovered'] / world_details['Confirmed'],2)
world_details['Death Rate'] = round(world_details['Deaths'] /world_details['Confirmed'], 2)
world_details = world_details.sort_values(by='Confirmed', ascending= False)
world_details.style.background_gradient(cmap='YlOrRd')
#asean_cases
countries = ['Vietnam', 'Thailand', 'Indonesia', 'Malaysia', 'Singapore', 'Philippines', 'Cambodia', 'Brunei']
asean = world_details.loc[countries]
asean = asean.sort_values(by='Confirmed', ascending=False)[:10]
asean.style.background_gradient(cmap='YlOrRd')
countries = ['Vietnam']
plot_time_variation_countries(data_select_agg, countries)
import datetime
import scipy
def plot_exponential_fit_data(d_df, title, delta):
    d_df = d_df.sort_values(by=['Date'], ascending=True)
    d_df['x'] = np.arange(len(d_df)) + 1
    d_df['y'] = d_df['Confirmed']

    x = d_df['x'][:-delta]
    y = d_df['y'][:-delta]

    c2 = scipy.optimize.curve_fit(lambda t,a,b: a*np.exp(b*t),  x,  y,  p0=(40, 0.1))
    #y = Ae^(Bx)
    A, B = c2[0]
    print(f'(y = Ae^(Bx)) A: {A}, B: {B}')
    x = range(1,d_df.shape[0] + 1)
    y_fit = A * np.exp(B * x)
    size = 3
    f, ax = plt.subplots(1,1, figsize=(4*size,2*size))
    g = sns.scatterplot(x=d_df['x'][:-delta], y=d_df['y'][:-delta], label='Confirmed cases (included for fit)', color='red')
    g = sns.scatterplot(x=d_df['x'][-delta:], y=d_df['y'][-delta:], label='Confirmed cases (validation)', color='blue')
    g = sns.lineplot(x=x, y=y_fit, label='Predicted values', color='green')
    plt.xlabel('Days since first case')
    plt.ylabel(f'cases')
    plt.title(f'Confirmed cases & predicted evolution: {title} \n Các ca dương tính Covid-19 và mô hình dự đoán')
    plt.xticks(rotation=90)
    ax.grid(color='black', linestyle='dotted', linewidth=0.75)
    plt.show()
data_ro = all_df[all_df['Country']=='Vietnam']
d_df = data_ro.copy()
plot_exponential_fit_data(d_df, 'Vietnam', 5)