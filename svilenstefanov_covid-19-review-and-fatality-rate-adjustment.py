# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input daty_chinaa files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import numpy as np
import pandas as pd

import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots
pio.templates.default = "plotly"
import datetime

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
data = pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv", parse_dates = ['ObservationDate'])
data.head()
data.info()
data.describe()
data['ObservationDate'] = data['ObservationDate'].apply(lambda x: x.date())
data = data.sort_values(by=['ObservationDate','Country/Region'])
data.rename(columns={'ObservationDate':'Date', 'Province/State':'Province', 'Country/Region':'Country', 'Deaths':'Death'}, inplace=True)
data['Active'] = data.Confirmed - data.Death - data.Recovered
data.head()
print(f"First entry: {data.Date.min()}")
print(f"Last entry: {data.Date.max()}")
print(f"Time length: {data.Date.max() - data.Date.min()}")
features = ['Confirmed', 'Active', 'Death', 'Recovered']
totals = data.groupby('Date')[features].sum().reset_index()
fig = go.Figure().add_trace(go.Scatter(x=totals.Date, y=totals.Confirmed, mode='lines', name='Confirmed'))\
    .add_trace(go.Scatter(x=totals.Date, y=totals.Death, mode='lines', name='Deaths'))\
    .add_trace(go.Scatter(x=totals.Date, y=totals.Recovered, mode='lines', name='Recovered'))\
    .add_trace(go.Scatter(x=totals.Date, y=totals.Active, mode='lines', name='Active'))\
.update_layout(title='Cases over time')
fig.show()
data_nochina = data[data.Country!='Mainland China'].groupby('Date')[features].sum().reset_index()
go.Figure().add_scatter(x=data_nochina.Date, y=data_nochina.Death, mode='lines', name='Deaths', line=dict(color='black'))\
.add_scatter(x=data_nochina.Date, y=data_nochina.Recovered, mode='lines', name='Recovered', line=dict(color='green'))\
.update_layout(title='Deaths vs Recoveries (excl. China)', yaxis_type='log')
latest = data.loc[data.Date == max(data['Date']),:]
latestByCountry = latest.groupby('Country')[features].sum().reset_index()
for feature in features:
    px.bar(latestByCountry.sort_values(by=feature,ascending=True).tail(10).reset_index(), x=feature, y='Country',
           orientation='h', title='Top 10 '+feature+' cases')
data_noprovinces = data.groupby(['Country','Date'])[features].sum().reset_index()
for feature in features:
    px.line(data_noprovinces[data_noprovinces['Country'].isin(
                data_noprovinces.groupby('Country')[features].sum().reset_index().
                sort_values(by=feature,ascending=False).head(10).Country)],
        x='Date', y=feature, color='Country', title=feature + ' cases over time by country')
data_noprovinces['Daily new'] = data_noprovinces.groupby('Country').Confirmed.diff()
data_noprovinces['Daily death'] = data_noprovinces.groupby('Country').Death.diff()
data_noprovinces['Daily recovered'] = data_noprovinces.groupby('Country').Recovered.diff()
data_noprovinces['Death_rate'] = 100*data_noprovinces['Death'] / data_noprovinces['Confirmed']
data_noprovinces['Recovery_rate'] = 100*data_noprovinces['Recovered'] / data_noprovinces['Confirmed']
data_noprovinces['Active_rate'] = 100*data_noprovinces['Active'] / data_noprovinces['Confirmed']
from datetime import timedelta
from datetime import date

countries10plusdeaths = data_noprovinces.loc[
    data_noprovinces.Country.isin(
        data_noprovinces.loc[data_noprovinces.Death > 10, 'Country']) &  (data_noprovinces.Country != 'Diamond Princess'),'Country'].unique()
fatality_adj_list = []
for country in countries10plusdeaths:
    for i in reversed(range(21)):
        step = []
        step.append(country)
        step.append(data_noprovinces.loc[(data_noprovinces.Date == data_noprovinces.Date.max() - timedelta(days=i)) & (data_noprovinces.Country == country), 'Date'].values)
        step.append(data_noprovinces.loc[(data_noprovinces.Date == data_noprovinces.Date.max() - timedelta(days=i)) & (data_noprovinces.Country == country), 'Death'].replace(np.nan, 0).values) # was daily death
        step.append(data_noprovinces.loc[(data_noprovinces.Date == data_noprovinces.Date.max() - timedelta(days=i)) & (data_noprovinces.Country == country), 'Recovered'].replace(np.nan, 0).values) # was daily recovered
        step.append(data_noprovinces.loc[(data_noprovinces.Date == data_noprovinces.Date.max() - timedelta(days=14+i)) & (data_noprovinces.Country == country),'Confirmed'].replace(np.nan, 0).values) # was daily new
        step.append(data_noprovinces.loc[(data_noprovinces.Date == data_noprovinces.Date.max() - timedelta(days=7+i)) & (data_noprovinces.Country == country),'Confirmed'].replace(np.nan, 0).values)
        step.append(data_noprovinces.loc[(data_noprovinces.Date == data_noprovinces.Date.max() - timedelta(days=i)) & (data_noprovinces.Country == country),'Confirmed'].replace(np.nan, 0).values)
        fatality_adj_list.append(step)
for i in range(len(fatality_adj_list)):
    # convert numpy datetime64 to pandas timestamp
    fatality_adj_list[i][1] = np.append(fatality_adj_list[i][1], date.today())
    fatality_adj_list[i][1] = pd.Timestamp(fatality_adj_list[i][1][0])
    # add 0 to all 4th element arrays, then get their first sub-element - this is a workaround to avoid the issue when a country didn't have data so far back
    for j in range(2,7):
        fatality_adj_list[i][j] = np.append(fatality_adj_list[i][j],0)
        fatality_adj_list[i][j] = fatality_adj_list[i][j][0]
        fatality_adj_list[i][j] = int(fatality_adj_list[i][j])

fatality_adj = pd.DataFrame(fatality_adj_list, columns=['Country', 'Date', 'Deaths', 'Recovered', 'Confirmed14daysago',
                                                        'Confirmed7daysago', 'ConfirmedToday'])
fatality_adj['Fatality Rate (Basic)'] = round(100*fatality_adj.Deaths / fatality_adj.ConfirmedToday, 2)
#fatality_adj['Fatality Rate (Adjusted) 14 days'] = round(100*fatality_adj.Deaths / fatality_adj.Confirmed14daysago, 2)
fatality_adj['Fatality Rate (Adjusted) 7 days'] = round(100*fatality_adj.Deaths / fatality_adj.Confirmed7daysago, 2).replace(np.nan,0).replace(np.inf,0)
fatality_adj['Recovery Rate (Adjusted) 7 days'] = round(100*fatality_adj.Recovered / fatality_adj.Confirmed7daysago, 2).replace(np.nan,0).replace(np.inf,0)
fatality_adj.Date = fatality_adj.Date.apply(lambda x: x.date())
fatality_adj = fatality_adj.sort_values(by=['Date','Country'])
fatality_adj.replace(np.nan,0,inplace=True)
latest_adj = fatality_adj[fatality_adj.Date == fatality_adj[fatality_adj.Country != 'Others'].Date.max()]
avg_fatality = round(100*latest_adj.Deaths.sum() / latest_adj.Confirmed7daysago.sum(),2)

print(f"The world's average fatality rate by yesterday is: {avg_fatality}%.")

temp = []
for i in range(10):
    temp.append(avg_fatality)
    
px.bar(latest_adj.sort_values(by='Fatality Rate (Adjusted) 7 days', ascending=False).head(10),
       x='Country', y='Fatality Rate (Adjusted) 7 days', color='Deaths', color_continuous_scale='Burg')\
.add_scatter(x=latest_adj.sort_values(by='Fatality Rate (Adjusted) 7 days', ascending=False).head(10).Country,
           y=temp, mode='lines', hovertemplate=f'{avg_fatality}%', name='World average',
           line=dict(color='black', dash='dash'), showlegend=False)\
.update_layout(title='Top 10 countries with highest fatality rate by yesterday')
print(f"The world's average fatality rate by yesterday is: {avg_fatality}%.")

px.bar(latest_adj[(latest_adj.Deaths>0)].sort_values(by='Fatality Rate (Adjusted) 7 days').head(10),
       x='Country', y='Fatality Rate (Adjusted) 7 days', text='Confirmed7daysago', color='Deaths', color_continuous_scale='Blugrn')\
.add_scatter(x=latest_adj[(latest_adj.Deaths>0)].sort_values(by='Fatality Rate (Adjusted) 7 days').head(10).Country,
           y=temp, mode='lines', hovertemplate=f'{avg_fatality}%', name='World average',
           line=dict(color='black', dash='dash'), showlegend=False)\
.update_layout(title='Top 10 countries with lowest fatality rate by yesterday')
avg_recovery = round(100*latest_adj.Recovered.sum() / latest_adj.Confirmed7daysago.sum(),2)
print(f"The world's average recovery rate by yesterday is: {avg_recovery}%.")

temp = []
for i in range(10):
    temp.append(avg_recovery)
    
px.bar(latest_adj[(latest_adj.Recovered>0)].sort_values(by='Recovery Rate (Adjusted) 7 days', ascending=False).head(10),
       x='Country', y='Recovery Rate (Adjusted) 7 days', text='Confirmed7daysago', color='Recovered', color_continuous_scale='Blugrn')\
.add_scatter(x=latest_adj[(latest_adj.Recovered>0)].sort_values(by='Recovery Rate (Adjusted) 7 days', ascending=False).head(10).Country,
           y=temp, mode='lines', hovertemplate=f'{avg_recovery}%', name='World average',
           line=dict(color='black', dash='dash'), showlegend=False)\
.update_layout(title='Top 10 countries with highest recovery rate by yesterday', yaxis_type='log')
print(f"The world's average recovery rate by yesterday is: {avg_recovery}%.")

px.bar(latest_adj[(latest_adj.Recovered>0)].sort_values(by='Recovery Rate (Adjusted) 7 days').head(10),
       x='Country', y='Recovery Rate (Adjusted) 7 days', text='Confirmed7daysago', color='Recovered', color_continuous_scale='Burg')\
.add_scatter(x=latest_adj[(latest_adj.Recovered>0)].sort_values(by='Recovery Rate (Adjusted) 7 days').head(10).Country,
           y=temp, mode='lines', hovertemplate=f'{avg_recovery}%', name='World average',
           line=dict(color='black', dash='dash'), showlegend=False)\
.update_layout(title='Top 10 countries with lowest recovery rate by yesterday')
totals_adj = fatality_adj.groupby('Date')['Confirmed7daysago', 'ConfirmedToday', 'Deaths', 'Recovered'].sum()
totals_adj['Fatality rate'] = round(100*totals_adj.Deaths / totals_adj.Confirmed7daysago,2)
totals_adj['Fatality rate (WHO methodology)'] = round(100*totals_adj.Deaths / totals_adj.ConfirmedToday,2)
totals_adj['Recovery rate'] = round(100*totals_adj.Recovered / totals_adj.Confirmed7daysago,2)
totals_adj['Recovery rate (WHO methodology)'] = round(100*totals_adj.Recovered / totals_adj.ConfirmedToday,2)
median_fatality_adj = []
for i in range(len(totals_adj)):
    median_fatality_adj.append(totals_adj['Fatality rate'].median())

median_fatality_who = []
for i in range(len(totals_adj)):
    median_fatality_who.append(totals_adj['Fatality rate (WHO methodology)'].median())
go.Figure().add_scatter(x=totals_adj.index, y=totals_adj['Fatality rate'], name='Fatality rate (adj)', line_shape='spline', mode='lines')\
.add_scatter(x=totals_adj.index, y=totals_adj['Fatality rate (WHO methodology)'], name='Fatality rate (WHO)', line_shape='spline', mode='lines')\
.add_scatter(x=totals_adj.index, y=median_fatality_adj, line=dict(dash='dash', color='royalblue'), name='Median fatality rate (adj)')\
.add_scatter(x=totals_adj.index, y=median_fatality_who, line=dict(dash='dash', color = 'firebrick'), name='Median fatality rate (WHO)')\
.update_layout(title='Fatality rate over time - adjusted vs. WHO methodology')
median_recovery_adj = []
for i in range(len(totals_adj)):
    median_recovery_adj.append(totals_adj['Recovery rate'].median())

median_recovery_who = []
for i in range(len(totals_adj)):
    median_recovery_who.append(totals_adj['Recovery rate (WHO methodology)'].median())
go.Figure().add_scatter(x=totals_adj.index, y=totals_adj['Recovery rate'], name='Recovery rate (adj)', line_shape='spline', mode='lines', line=dict(color='green'))\
.add_scatter(x=totals_adj.index, y=totals_adj['Recovery rate (WHO methodology)'], name='Recovery rate (WHO)', line_shape='spline', line=dict(color='yellow'))\
.add_scatter(x=totals_adj.index, y=median_recovery_adj, name='Median recovery rate (adj)', line=dict(dash='dash', color='green'))\
.add_scatter(x=totals_adj.index, y=median_recovery_who, name='Median recovery rate (WHO)', line=dict(dash='dash', color='yellow'))\
.update_layout(title='Recovery rate over time - adjusted vs. WHO methodology')
top10_fatality_median = fatality_adj.groupby('Country')['Fatality Rate (Adjusted) 7 days'].median().sort_values(ascending=False).head(10).reset_index().Country
top10_recovery_median = fatality_adj.groupby('Country')['Recovery Rate (Adjusted) 7 days'].median().sort_values(ascending=False).head(10).reset_index().Country
fatality_adj.loc[fatality_adj['Fatality Rate (Adjusted) 7 days'] > 100, 'Fatality Rate (Adjusted) 7 days'] = 100
fatality_adj.loc[fatality_adj['Recovery Rate (Adjusted) 7 days'] > 100, 'Recovery Rate (Adjusted) 7 days'] = 100
fatality_adj.loc[fatality_adj['Fatality Rate (Adjusted) 7 days'] < 0, 'Fatality Rate (Adjusted) 7 days'] = 0
fatality_adj.loc[fatality_adj['Recovery Rate (Adjusted) 7 days'] < 0, 'Recovery Rate (Adjusted) 7 days'] = 0
rates = ['Fatality Rate (Adjusted) 7 days', 'Recovery Rate (Adjusted) 7 days']

for rate in rates:
    px.line(fatality_adj[fatality_adj['Country'].isin(top10_fatality_median)],
        x='Date', y=rate, color='Country', title = (rate[:14] .replace('Rate','rate') + 'over time by country (Top 10 median)'),
           line_shape='spline')
for feature in features:
    px.choropleth(latestByCountry, locations = 'Country', locationmode = 'country names', color=feature,
             color_continuous_scale = 'Portland', title = 'World map of ' + feature + ' cases',
             range_color = [1,1000 if feature!='Confirmed' else 2000])
data_noprovinces['Daily new pct'] = data_noprovinces.groupby('Country').Confirmed.pct_change()*100
data_noprovinces['Daily death pct'] = data_noprovinces.groupby('Country').Death.pct_change()*100
data_noprovinces['Daily recovered pct'] = data_noprovinces.groupby('Country').Recovered.pct_change()*100
px.bar(data_noprovinces[data_noprovinces.Date == data_noprovinces.Date.max()].sort_values(by='Daily new', ascending=False).head(10),
       x='Country', y='Daily new', title="Last day's Top 10 by number of new confirmed cases", color='Confirmed', color_continuous_scale='Portland')\
.update_layout(legend_title='<b>Total Confirmed</b>')
px.bar(data_noprovinces[data_noprovinces.Date == data_noprovinces.Date.max()].sort_values(by='Daily death', ascending=False).head(10),
       x='Country', y='Daily death', title="Last day's Top 10 by number of new death cases", color='Death', color_continuous_scale='matter')\
.update_layout(legend_title='Total Deaths>')
px.bar(data_noprovinces[data_noprovinces.Date == data_noprovinces.Date.max()].sort_values(by='Daily recovered', ascending=False).head(10),
       x='Country', y='Daily recovered', title="Last day's Top 10 by number of new recovered cases", color='Recovered', color_continuous_scale='RdBu')
top10conf = data_noprovinces.groupby('Country')['Confirmed'].sum().reset_index().sort_values(by='Confirmed',ascending=False).head(10).Country
top10deaths = data_noprovinces.groupby('Country')['Death'].sum().reset_index().sort_values(by='Death',ascending=False).head(10).Country
top10recov = data_noprovinces.groupby('Country')['Recovered'].sum().reset_index().sort_values(by='Recovered',ascending=False).head(10).Country
top10active = data_noprovinces.groupby('Country')['Active'].sum().reset_index().sort_values(by='Active',ascending=False).head(10).Country
px.line(data_noprovinces[data_noprovinces.Country.isin(top10conf)], x='Date', y='Daily new', color='Country', title='Daily new cases', line_shape='spline')
px.line(data_noprovinces[data_noprovinces.Country.isin(top10deaths)], x='Date', y='Daily death', color='Country', title='Daily death cases', line_shape='spline')
px.line(data_noprovinces[data_noprovinces.Country.isin(top10recov)], x='Date', y='Daily recovered', color='Country', title='Daily recovered cases', line_shape='spline')
last7days = data_noprovinces.sort_values(by=['Date','Country'], ascending=True).tail(len(data_noprovinces.Country.unique())*7)
top10highConf7days = last7days[last7days.Confirmed > 500].groupby('Country')['Daily new pct'].median().reset_index().replace(np.inf,0).sort_values(by='Daily new pct', ascending=False).head(10).Country
top10highDeath7days = last7days[last7days.Death > 50].groupby('Country')['Daily death pct'].median().reset_index().replace(np.inf,0).replace(np.inf,0).sort_values(by='Daily death pct', ascending=False).head(10).Country
top10highRecov7days = last7days[last7days.Recovered > 50].groupby('Country')['Daily recovered pct'].median().reset_index().replace(np.inf,0).replace(np.inf,0).sort_values(by='Daily recovered pct', ascending=False).head(10).Country
px.line(data_noprovinces.loc[(data_noprovinces.Country.isin(top10highConf7days)) & (data_noprovinces.Date >= data.Date.max()-timedelta(days=7))], x='Date', y='Daily new', color='Country',
        title='Top 10 countries with highest average daily growth of Confirmed new cases for the last 7 days', line_shape='spline')\
.update_layout(yaxis_type='log')

px.bar(last7days[last7days.Confirmed > 500].groupby('Country')['Daily new pct'].median().reset_index().replace(np.inf,0).sort_values(by='Daily new pct', ascending=False).head(10),
      x='Country', y='Daily new pct', title='Top 10 countries with highest average daily growth of Confirmed new cases for the last 7 days')
px.line(data_noprovinces[(data_noprovinces.Country.isin(top10highDeath7days)) & (data_noprovinces.Date >= data.Date.max()-timedelta(days=7))],
        x='Date', y='Daily death pct', color='Country', title='Top 10 countries with highest median daily growth of new Death cases for the last 7 days', line_shape='spline')#\
#.update_layout(yaxis_type='log')

px.bar(last7days[last7days.Death > 50].groupby('Country')['Daily death pct'].median().reset_index().replace(np.inf,0).replace(np.inf,0).sort_values(by='Daily death pct', ascending=False).head(10),
      x='Country', y='Daily death pct', title='Top 10 countries with highest median daily growth of new Death cases for the last 7 days')
px.line(data_noprovinces[data_noprovinces.Country.isin(top10highRecov7days) & (data_noprovinces.Date >= data.Date.max()-timedelta(days=7))],
        x='Date', y='Daily recovered pct', color='Country', title='Top 10 countries with highest median daily growth of Recoveries for the last 7 days', line_shape='spline')
px.bar(last7days[last7days.Recovered > 50].groupby('Country')['Daily recovered pct'].mean().reset_index().replace(np.inf,0).replace(np.inf,0).sort_values(by='Daily recovered pct', ascending=False).head(10),
       x='Country', y='Daily recovered pct', title='Top 10 countries with highest median daily growth of Recoveries for the last 7 days')
tests_input = pd.read_csv("../input/countryinfo/covid19countryinfo.csv")
tests_input.head()
tests_input.describe()
tests = tests_input[tests_input.tests.isna() == False]
tests = tests.loc[:,['country','pop','tests','density','medianage', 'urbanpop', 'gatheringlimit', 'hospibed', 'smokers', 'lung', 'healthexp', 'fertility']]
tests = tests.rename(columns={'country': 'Country', 'pop': 'Population', 'density': 'Density', 'medianage': 'Median Age', 'tests': 'Tests'})
continents = pd.read_csv("../input/country-to-continent/countryContinent.csv", encoding='latin-1')
continents.head()
continents = continents.loc[:,['country','continent','sub_region']]
continents = continents.rename(columns={'country':'Country', 'continent':'Continent', 'sub_region':'Region'})
tests_merged = pd.merge(latestByCountry, tests, on='Country')
tests_merged.Population = tests_merged.Population.apply(lambda x: int(x.replace(',','')))
tests_merged = pd.merge(tests_merged, continents, on='Country', how='left')
tests_merged[tests_merged.Continent.isna()]
tests_merged.loc[tests_merged.Country=='Russia','Continent'] = 'Europe'
tests_merged.loc[tests_merged.Country=='Russia','Region'] = 'Eastern Europe'
tests_merged.loc[tests_merged.Country=='US','Continent'] = 'Americas'
tests_merged.loc[tests_merged.Country=='US','Region'] = 'Northern America'
tests_merged.loc[tests_merged.Country=='Vietnam','Continent'] = 'Asia'
tests_merged.loc[tests_merged.Country=='Vietnam','Region'] = 'South-Eastern Asia'
tests_merged['Tests1m'] = round(1000000*tests_merged.Tests/tests_merged.Population,2)
tests_merged['Confirmed1m'] = round(1000000*tests_merged.Confirmed/tests_merged.Population,2)
tests_merged['Deaths1m'] = round(1000000*tests_merged.Death/tests_merged.Population,2)
tests_merged['Recovered1m'] = round(1000000*tests_merged.Recovered/tests_merged.Population,2)
corrmatrix=tests_merged.corr()

go.Figure(data=go.Heatmap(z=corrmatrix, x=corrmatrix.index, y=corrmatrix.columns))\
.update_layout(title='Tests Heatmap')
px.scatter(x=tests_merged.Tests1m, y=tests_merged.Confirmed1m, size=tests_merged['Median Age'], text=tests_merged.Country, color=tests_merged['Continent'])\
.update_traces(textposition='top center', textfont_size=10)\
.update_layout(title='Tests conducted vs Confirmed cases by country', xaxis_type='log', yaxis_type='log', xaxis_title='Tests per 1 million population', yaxis_title='Confirmed cases per 1 million population')
px.scatter(x=tests_merged.Tests1m, y=tests_merged.Deaths1m, size=tests_merged['Median Age']-22, text=tests_merged.Country, color=tests_merged['Continent'])\
.update_traces(textposition='top center', textfont_size=10)\
.update_layout(title='Tests conducted vs Deaths by country', xaxis_type='log', yaxis_type='log', xaxis_title='Tests per 1 million population', yaxis_title='Deaths per 1 million population')
px.scatter(x=tests_merged.Tests1m, y=tests_merged.Recovered1m, size=tests_merged['Median Age']-22, text=tests_merged.Country, color=tests_merged['Continent'])\
.update_traces(textposition='top center', textfont_size=10)\
.update_layout(title='Tests conducted vs Recoveries by country', xaxis_type='log', yaxis_type='log', xaxis_title='Tests per 1 million population', yaxis_title='Recoveries per 1 million population')
restrictions = tests_input.loc[tests_input.quarantine.isna() == False,['country','quarantine','schools', 'publicplace', 'gathering', 'nonessential']]
restrictions = restrictions.groupby('country').first().reset_index()
def to_date(x):
    converttodate = datetime.date(int(x.split('/')[2]), int(x.split('/')[0]), int(x.split('/')[1]))
    return converttodate

restrictions.quarantine = restrictions.quarantine.apply(lambda x: to_date(x))
restrictions.groupby(['quarantine','country']).count()
go.Figure()\
.add_scatter(x=data_noprovinces.loc[(data_noprovinces.Country == 'Italy') & (data_noprovinces.Date < (restrictions.loc[restrictions.country=='Italy','quarantine'].values[0])),'Date'],
                        y=data_noprovinces.loc[(data_noprovinces.Country == 'Italy') & (data_noprovinces.Date < (restrictions.loc[restrictions.country=='Italy','quarantine'].values[0])),'Confirmed'], mode='lines', name='Italy before')\
.add_scatter(x=data_noprovinces.loc[(data_noprovinces.Country == 'Italy') & (data_noprovinces.Date >= (restrictions.loc[restrictions.country=='Italy','quarantine'].values[0])),'Date'],
                        y=data_noprovinces.loc[(data_noprovinces.Country == 'Italy') & (data_noprovinces.Date >= (restrictions.loc[restrictions.country=='Italy','quarantine'].values[0])),'Confirmed'], mode='lines', name='Italy after')\
.add_scatter(x=data_noprovinces.loc[(data_noprovinces.Country == 'Germany') & (data_noprovinces.Date < (restrictions.loc[restrictions.country=='Germany','quarantine'].values[0])),'Date'],
                        y=data_noprovinces.loc[(data_noprovinces.Country == 'Germany') & (data_noprovinces.Date < (restrictions.loc[restrictions.country=='Germany','quarantine'].values[0])),'Confirmed'], mode='lines', name='Germany before')\
.add_scatter(x=data_noprovinces.loc[(data_noprovinces.Country == 'Germany') & (data_noprovinces.Date >= (restrictions.loc[restrictions.country=='Germany','quarantine'].values[0])),'Date'],
                        y=data_noprovinces.loc[(data_noprovinces.Country == 'Germany') & (data_noprovinces.Date >= (restrictions.loc[restrictions.country=='Germany','quarantine'].values[0])),'Confirmed'], mode='lines', name='Germany after')\
.add_scatter(x=data_noprovinces.loc[(data_noprovinces.Country == 'France') & (data_noprovinces.Date < (restrictions.loc[restrictions.country=='France','quarantine'].values[0])),'Date'],
                        y=data_noprovinces.loc[(data_noprovinces.Country == 'France') & (data_noprovinces.Date < (restrictions.loc[restrictions.country=='France','quarantine'].values[0])),'Confirmed'], mode='lines', name='France before')\
.add_scatter(x=data_noprovinces.loc[(data_noprovinces.Country == 'France') & (data_noprovinces.Date >= (restrictions.loc[restrictions.country=='France','quarantine'].values[0])),'Date'],
                        y=data_noprovinces.loc[(data_noprovinces.Country == 'France') & (data_noprovinces.Date >= (restrictions.loc[restrictions.country=='France','quarantine'].values[0])),'Confirmed'], mode='lines', name='France after')\
.add_scatter(x=data_noprovinces.loc[(data_noprovinces.Country == 'Spain') & (data_noprovinces.Date < (restrictions.loc[restrictions.country=='Spain','quarantine'].values[0])),'Date'],
                        y=data_noprovinces.loc[(data_noprovinces.Country == 'Spain') & (data_noprovinces.Date < (restrictions.loc[restrictions.country=='Spain','quarantine'].values[0])),'Confirmed'], mode='lines', name='Spain before')\
.add_scatter(x=data_noprovinces.loc[(data_noprovinces.Country == 'Spain') & (data_noprovinces.Date >= (restrictions.loc[restrictions.country=='Spain','quarantine'].values[0])),'Date'],
                        y=data_noprovinces.loc[(data_noprovinces.Country == 'Spain') & (data_noprovinces.Date >= (restrictions.loc[restrictions.country=='Spain','quarantine'].values[0])),'Confirmed'], mode='lines', name='Spain after')\
.add_scatter(x=data_noprovinces.loc[(data_noprovinces.Country == 'Belgium') & (data_noprovinces.Date < (restrictions.loc[restrictions.country=='Belgium','quarantine'].values[0])),'Date'],
                        y=data_noprovinces.loc[(data_noprovinces.Country == 'Belgium') & (data_noprovinces.Date < (restrictions.loc[restrictions.country=='Belgium','quarantine'].values[0])),'Confirmed'], mode='lines', name='Belgium before')\
.add_scatter(x=data_noprovinces.loc[(data_noprovinces.Country == 'Belgium') & (data_noprovinces.Date >= (restrictions.loc[restrictions.country=='Belgium','quarantine'].values[0])),'Date'],
                        y=data_noprovinces.loc[(data_noprovinces.Country == 'Belgium') & (data_noprovinces.Date >= (restrictions.loc[restrictions.country=='Belgium','quarantine'].values[0])),'Confirmed'], mode='lines', name='Belgium after')\
.add_scatter(x=data_noprovinces.loc[(data_noprovinces.Country == 'Iran') & (data_noprovinces.Date < (restrictions.loc[restrictions.country=='Iran','quarantine'].values[0])),'Date'],
                        y=data_noprovinces.loc[(data_noprovinces.Country == 'Iran') & (data_noprovinces.Date < (restrictions.loc[restrictions.country=='Iran','quarantine'].values[0])),'Confirmed'], mode='lines', name='Iran before')\
.add_scatter(x=data_noprovinces.loc[(data_noprovinces.Country == 'Iran') & (data_noprovinces.Date >= (restrictions.loc[restrictions.country=='Iran','quarantine'].values[0])),'Date'],
                        y=data_noprovinces.loc[(data_noprovinces.Country == 'Iran') & (data_noprovinces.Date >= (restrictions.loc[restrictions.country=='Iran','quarantine'].values[0])),'Confirmed'], mode='lines', name='Iran after')\
.add_scatter(x=data_noprovinces.loc[(data_noprovinces.Country == 'Bulgaria') & (data_noprovinces.Date < (restrictions.loc[restrictions.country=='Iran','quarantine'].values[0])),'Date'],
                        y=data_noprovinces.loc[(data_noprovinces.Country == 'Bulgaria') & (data_noprovinces.Date < (restrictions.loc[restrictions.country=='Iran','quarantine'].values[0])),'Confirmed'], mode='lines', name='Bulgaria before')\
.add_scatter(x=data_noprovinces.loc[(data_noprovinces.Country == 'Bulgaria') & (data_noprovinces.Date >= (restrictions.loc[restrictions.country=='Iran','quarantine'].values[0])),'Date'],
                        y=data_noprovinces.loc[(data_noprovinces.Country == 'Bulgaria') & (data_noprovinces.Date >= (restrictions.loc[restrictions.country=='Iran','quarantine'].values[0])),'Confirmed'], mode='lines', name='Bulgaria after')\
.update_layout(title='Confirmed cases before and after imposing national quarantine', yaxis_type='log')