# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
#import calmap
import folium
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/corona-virus-report/covid_19_clean_complete.csv",parse_dates=['Date'])
# cases 
cases = ['Confirmed', 'Deaths', 'Recovered', 'Still Infected']

# still infected = confirmed - deaths - recovered
data['Still Infected'] = data['Confirmed'] - data['Deaths'] - data['Recovered']

# replacing Mainland china with just China
data['Country/Region'] = data['Country/Region'].replace('Mainland China', 'China')

# filling missing values 
data[['Province/State']] = data[['Province/State']].fillna('NA')
data[cases] = data[cases].fillna(0)
temp = data.groupby('Date')['Confirmed', 'Deaths', 'Recovered', 'Still Infected'].sum().reset_index()
temp = temp[temp['Date']==max(temp['Date'])].reset_index(drop=True)
temp
tm = temp.melt(id_vars="Date", value_vars=['Still Infected', 'Deaths', 'Recovered'])
fig = px.treemap(tm, path=["variable"], values="value", height=200)
fig.show()
d = data.groupby(['Province/State','Country/Region'],as_index=False)['Province/State','Country/Region','Confirmed','Recovered','Deaths'].sum()
d["world"] = "world"
fig = px.treemap(d, path=['world', 'Country/Region', 'Province/State'], values='Confirmed',
                  color_continuous_scale='RdBu',
                  color_continuous_midpoint=np.average(d['Deaths'], weights=d['Confirmed']))
fig.show()
d["world"] = "world"
fig = px.treemap(d, path=['world', 'Country/Region', 'Province/State'], values='Deaths',
                  color_continuous_scale='RdBu',)
fig.show()
d['recover%'] = d['Recovered']/d['Confirmed']
d["world"] = "world"
fig = px.treemap(d, path=['world', 'Country/Region', 'Province/State'], values='recover%',
                  color_continuous_scale='RdBu',)
fig.show()
del d
del temp
# cases in the ships
ship = data[data['Province/State'].str.lower().str.contains('ship')]

# china and the row
china = data[data['Country/Region']=='China']
row = data[data['Country/Region']!='China']

# latest
data_latest = data[data['Date'] == max(data['Date'])].reset_index()
china_latest = data_latest[data_latest['Country/Region']=='China']
row_latest = data_latest[data_latest['Country/Region']!='China']

data_latestt_grouped_grouped = data_latest.groupby('Country/Region')['Confirmed', 'Deaths', 'Recovered', 'Still Infected'].sum().reset_index()
china_latest_grouped = china_latest.groupby('Province/State')['Confirmed', 'Deaths', 'Recovered', 'Still Infected'].sum().reset_index()
row_latest_grouped = row_latest.groupby('Country/Region')['Confirmed', 'Deaths', 'Recovered', 'Still Infected'].sum().reset_index()
temp = data.groupby('Date')['Still Infected', 'Deaths', 'Recovered'].sum().reset_index()
temp = temp.melt(id_vars="Date", value_vars=['Still Infected', 'Deaths', 'Recovered'],
                 var_name='Case', value_name='Count')
temp.head()

fig = px.bar(temp, x="Date", y="Count", color='Case',
             title='Cases over time')
fig.show()

temp = data.groupby(['Country/Region', 'Date'])['Confirmed', 'Deaths', 'Recovered'].sum()
temp = temp.reset_index()
# temp.head()

fig = px.bar(temp, x="Date", y="Confirmed", color='Country/Region', orientation='v', height=600,
             title='Confirmed', color_discrete_sequence = px.colors.cyclical.mygbm)
fig.show()

fig = px.bar(temp, x="Date", y="Deaths", color='Country/Region', orientation='v', height=600,
             title='Deaths', color_discrete_sequence = px.colors.cyclical.mygbm)
fig.show()
temp_f = data_latestt_grouped_grouped.sort_values(by='Confirmed', ascending=False)
temp_f = temp_f.reset_index(drop=True)
temp_f.style.background_gradient(cmap='YlOrRd')
temp_flg = data_latestt_grouped_grouped[['Country/Region', 'Deaths']]
temp_flg = temp_flg.sort_values(by='Deaths', ascending=False)
temp_flg = temp_flg.reset_index(drop=True)
temp_flg = temp_flg[temp_flg['Deaths']>0]
temp_flg.style.background_gradient(cmap='Reds')
temp = data_latestt_grouped_grouped[data_latestt_grouped_grouped['Recovered']==0]
temp = temp[['Country/Region', 'Confirmed', 'Deaths', 'Recovered']]
temp = temp.sort_values('Confirmed', ascending=False)
temp = temp.reset_index(drop=True)
temp.style.background_gradient(cmap='YlOrRd')
temp = row_latest_grouped[row_latest_grouped['Confirmed']==
                          row_latest_grouped['Deaths']]
temp = temp[['Country/Region', 'Confirmed', 'Deaths']]
temp = temp.sort_values('Confirmed', ascending=False)
temp = temp.reset_index(drop=True)
temp.style.background_gradient(cmap='Reds')
temp = row_latest_grouped[row_latest_grouped['Confirmed']==
                          row_latest_grouped['Recovered']]
temp = temp[['Country/Region', 'Confirmed', 'Recovered']]
temp = temp.sort_values('Confirmed', ascending=False)
temp = temp.reset_index(drop=True)
temp.style.background_gradient(cmap='Greens')
temp = row_latest_grouped[row_latest_grouped['Confirmed']==
                          row_latest_grouped['Deaths']+
                          row_latest_grouped['Recovered']]
temp = temp[['Country/Region', 'Confirmed', 'Deaths', 'Recovered']]
temp = temp.sort_values('Confirmed', ascending=False)
temp = temp.reset_index(drop=True)
temp.style.background_gradient(cmap='Greens')
# World wide

m = folium.Map(location=[0, 0], tiles='cartodbpositron',
               min_zoom=1, max_zoom=4, zoom_start=1)

for i in range(0, len(data_latest)):
    folium.Circle(
        location=[data_latest.iloc[i]['Lat'], data_latest.iloc[i]['Long']],
        color='crimson', 
        tooltip =   '<li><bold>Country : '+str(data_latest.iloc[i]['Country/Region'])+
                    '<li><bold>Province : '+str(data_latest.iloc[i]['Province/State'])+
                    '<li><bold>Confirmed : '+str(data_latest.iloc[i]['Confirmed'])+
                    '<li><bold>Deaths : '+str(data_latest.iloc[i]['Deaths'])+
                    '<li><bold>Recovered : '+str(data_latest.iloc[i]['Recovered']),
        radius=int(data_latest.iloc[i]['Confirmed'])**1.1).add_to(m)
m
fig = px.choropleth(data_latestt_grouped_grouped, locations="Country/Region", 
                    locationmode='country names', color="Confirmed", 
                    hover_name="Country/Region", range_color=[1,7000], 
                    color_continuous_scale="agsunset", 
                    title='Countries with Confirmed Cases')
fig.update(layout_coloraxis_showscale=False)
fig.show()

# ------------------------------------------------------------------------

fig = px.choropleth(data_latestt_grouped_grouped[data_latestt_grouped_grouped['Deaths']>0], 
                    locations="Country/Region", locationmode='country names',
                    color="Deaths", hover_name="Country/Region", 
                    range_color=[1,50], color_continuous_scale="agsunset",
                    title='Countries with Deaths Reported')
fig.update(layout_coloraxis_showscale=False)
fig.show()
fig = px.bar(data_latestt_grouped_grouped[['Country/Region', 'Confirmed']].sort_values('Confirmed', ascending=False), 
             x="Confirmed", y="Country/Region", color='Country/Region', orientation='h',
             log_x=True, color_discrete_sequence = px.colors.qualitative.Bold, title='Confirmed Cases', height=1200)
fig.show()

temp = data_latestt_grouped_grouped[['Country/Region', 'Deaths']].sort_values('Deaths', ascending=False)
fig = px.bar(temp[temp['Deaths']>0], 
             x="Deaths", y="Country/Region", color='Country/Region', title='Deaths', orientation='h',
             log_x=True, color_discrete_sequence = px.colors.qualitative.Bold)
fig.show()
# In China
temp = china.groupby('Date')['Confirmed', 'Deaths', 'Recovered'].sum().diff()
temp = temp.reset_index()
temp = temp.melt(id_vars="Date", 
                 value_vars=['Confirmed', 'Deaths', 'Recovered'])

fig = px.line(temp, x="Date", y="value", color='variable', 
             title='In China')
fig.update_layout(barmode='group')
fig.show()

#-----------------------------------------------------------------------------

# ROW
temp = row.groupby('Date')['Confirmed', 'Deaths', 'Recovered'].sum().diff()
temp = temp.reset_index()
temp = temp.melt(id_vars="Date", 
                 value_vars=['Confirmed', 'Deaths', 'Recovered'])

fig = px.line(temp, x="Date", y="value", color='variable', 
             title='Outside China')
fig.update_layout(barmode='group')
fig.show()
def from_china_or_not(row):
    if row['Country/Region']=='China':
        return 'From China'
    else:
        return 'Outside China'
    
temp = data.copy()
temp['Region'] = temp.apply(from_china_or_not, axis=1)
temp = temp.groupby(['Region', 'Date'])['Confirmed', 'Deaths', 'Recovered']
temp = temp.sum().diff().reset_index()
mask = temp['Region'] != temp['Region'].shift(1)
temp.loc[mask, 'Confirmed'] = np.nan
temp.loc[mask, 'Deaths'] = np.nan
temp.loc[mask, 'Recovered'] = np.nan

fig = px.bar(temp, x='Date', y='Confirmed', color='Region', barmode='group', 
             text='Confirmed', title='Confirmed')
fig.update_traces(textposition='outside')
fig.show()

fig = px.bar(temp, x='Date', y='Deaths', color='Region', barmode='group', 
             text='Confirmed', title='Deaths')
fig.update_traces(textposition='outside')
fig.update_traces(textangle=-90)
fig.show()
temp = data.groupby(['Country/Region', 'Date', ])['Confirmed', 'Deaths', 'Recovered']
temp = temp.sum().diff().reset_index()

mask = temp['Country/Region'] != temp['Country/Region'].shift(1)

temp.loc[mask, 'Confirmed'] = np.nan
temp.loc[mask, 'Deaths'] = np.nan
temp.loc[mask, 'Recovered'] = np.nan

fig = px.bar(temp, x="Date", y="Confirmed", color='Country/Region',
             title='Number of new cases everyday')
fig.show()
fig = px.bar(temp[temp['Country/Region']!='China'], x="Date", y="Confirmed", color='Country/Region',
             title='Number of new cases outside China everyday')
fig.show()
fig = px.bar(temp, x="Date", y="Deaths", color='Country/Region',
             title='Number of new death case reported outside China everyday')
fig.show()
fig = px.bar(temp[temp['Country/Region']!='China'], x="Date", y="Deaths", color='Country/Region',
             title='Number of new death case reported outside China everyday')
fig.show()
c_spread = china[china['Confirmed']!=0].groupby('Date')['Province/State'].unique().apply(len)
c_spread = pd.DataFrame(c_spread).reset_index()

fig = px.line(c_spread, x='Date', y='Province/State', 
              title='Number of Provinces/States/Regions of China to which COVID-19 spread over the time')
fig.show()
spread = data[data['Confirmed']!=0].groupby('Date')['Country/Region'].unique().apply(len)
spread = pd.DataFrame(spread).reset_index()

fig = px.line(spread, x='Date', y='Country/Region', 
              title='Number of Countries/Regions to which COVID-19 spread over the time')
fig.show()