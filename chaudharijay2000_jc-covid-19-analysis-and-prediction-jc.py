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
!pip install calmap
import random
from urllib.request import urlopen

# storing and anaysis
import numpy as np
import pandas as pd

# visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
import plotly.figure_factory as ff
import folium
import calmap

# converter
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()   

# hide warnings
import warnings
warnings.filterwarnings('ignore')

# html embedding
from IPython.display import Javascript
from IPython.core.display import display
from IPython.core.display import HTML

covid_df = pd.read_csv('/kaggle/input/corona-virus-report/covid_19_clean_complete.csv', parse_dates=['Date'])
covid_df.head()
covid_df.dtypes
print('Number of data points : ', covid_df.shape[0])
print('Number of features : ', covid_df.shape[1])
#to indicate if any value is missing. Any missing values?
covid_df.isnull().values.any()
# Total missing values for each feature
covid_df.isnull().sum()
# replacing state missing values by "unknow"
covid_df['Province/State'] = covid_df['Province/State'].fillna('unknown')
# Replace with Mainland China to China
covid_df.replace('Mainland China', 'China', inplace = True)
covid_df.isnull().values.any()
import pandas_profiling
pandas_profiling.ProfileReport(covid_df)
cases = ['Confirmed', 'Deaths', 'Recovered', 'Active']
# Active Case = confirmed - deaths - recovered
covid_df['Active'] = covid_df['Confirmed'] - covid_df['Deaths'] - covid_df['Recovered']
covid_df[cases] = covid_df[cases].fillna(0)
# latest data
covid_latest = covid_df[covid_df['Date'] == max(covid_df['Date'])].reset_index()
china_latest = covid_latest[covid_latest['Country/Region']=='China']
row_latest = covid_latest[covid_latest['Country/Region']!='China']
# condensed latest data
covid_latest_grouped = covid_latest.groupby('Country/Region')['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()
china_latest_grouped = china_latest.groupby('Province/State')['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()
row_latest_grouped = row_latest.groupby('Country/Region')['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()

covid = covid_df.groupby(['Country/Region', 'Province/State'])['Confirmed', 'Deaths', 'Recovered', 'Active'].max()
covid = covid_df.groupby('Date')['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()
covid = covid[covid['Date']==max(covid['Date'])].reset_index(drop=True)
covid.style.background_gradient(cmap='viridis')
temp_f = covid_latest_grouped.sort_values(by='Confirmed', ascending=False)
temp_f = temp_f.reset_index(drop=True)
temp_f.style.background_gradient(cmap='Reds')
temp = temp_f[temp_f['Recovered']==0][['Country/Region', 'Confirmed', 'Deaths', 'Recovered']]
temp.reset_index(drop=True).style.background_gradient(cmap='Reds')
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
temp_f = china_latest_grouped[['Province/State', 'Confirmed', 'Deaths', 'Recovered']]
temp_f = temp_f.sort_values(by='Confirmed', ascending=False)
temp_f = temp_f.reset_index(drop=True)
temp_f.style.background_gradient(cmap='Pastel1_r')
temp = china_latest_grouped[china_latest_grouped['Confirmed']==
                          china_latest_grouped['Recovered']]
temp = temp[['Province/State', 'Confirmed','Deaths', 'Recovered']]
temp = temp.sort_values('Confirmed', ascending=False)
temp = temp.reset_index(drop=True)
temp.style.background_gradient(cmap='Greens')
# Ploting cases on world map
import folium
import math
world_map = covid_df[covid_df['Date'] == covid_df['Date'].max()]
map = folium.Map(location=[30, 30], tiles = "cartodbpositron", zoom_start=2.2)
for i in range(0,len(world_map)):
    folium.Circle(location=[world_map.iloc[i]['Lat'],
                            world_map.iloc[i]['Long']],
                            radius=(math.sqrt(world_map.iloc[i]['Confirmed'])*4000 ),
                            color='crimson',
                            fill=True,
                            fill_color='crimson').add_to(map)
map
# https://plot.ly/python/choropleth-maps/
fig = px.choropleth(covid_latest_grouped, locations="Country/Region", 
                    locationmode='country names', color="Confirmed", 
                    hover_name="Country/Region", range_color=[1,7000], 
                    color_continuous_scale="aggrnyl", 
                    title='Countries with Confirmed Cases')
fig.update(layout_coloraxis_showscale=False)
fig.show()
# https://plot.ly/python/choropleth-maps/
import plotly.express as px

fig = px.choropleth(covid_latest_grouped, locations="Country/Region", locationmode='country names', 
                    color="Confirmed", hover_name="Country/Region", range_color=[1,50], color_continuous_scale="Sunsetdark", 
                    title='Countries with Confirmed Cases')
fig.update(layout_coloraxis_showscale=False)
fig.show()
fig = px.choropleth(covid_latest_grouped[covid_latest_grouped['Deaths']>0], locations="Country/Region", locationmode='country names',
                    color="Deaths", hover_name="Country/Region", range_color=[1,50], color_continuous_scale="Peach",
                    title='Countries with Deaths Reported')
fig.update(layout_coloraxis_showscale=False)
fig.show()
formated_gdf = covid_df.groupby(['Date', 'Country/Region'])['Confirmed', 'Deaths', 'Recovered'].max()
formated_gdf = formated_gdf.reset_index()
formated_gdf['Date'] = pd.to_datetime(formated_gdf['Date'])
formated_gdf['Date'] = formated_gdf['Date'].dt.strftime('%m/%d/%Y')
formated_gdf['size'] = formated_gdf['Confirmed'].pow(0.3)

fig = px.scatter_geo(formated_gdf, locations="Country/Region", locationmode='country names', 
                     color="Confirmed", size='size', hover_name="Country/Region", 
                     range_color= [0, max(formated_gdf['Confirmed'])+2], 
                     projection="natural earth", animation_frame="Date", 
                     title='Spread over time')
fig.update(layout_coloraxis_showscale=False)
fig.show()

flg = covid_latest_grouped
flg.head()
fig = px.bar(flg.sort_values('Confirmed', ascending=False).head(20).sort_values('Confirmed', ascending=True), 
             x="Confirmed", y="Country/Region", title='Confirmed Cases', text='Confirmed', orientation='h', 
             width=700, height=700, range_x = [0, max(flg['Confirmed'])+10000])
fig.update_traces(marker_color='#46cdcf', opacity=0.8, textposition='outside')
fig.show()
fig = px.bar(flg.sort_values('Deaths', ascending=False).head(20).sort_values('Deaths', ascending=True), 
             x="Deaths", y="Country/Region", title='Deaths', text='Deaths', orientation='h', 
             width=700, height=700, range_x = [0, max(flg['Deaths'])+500])
fig.update_traces(marker_color="#ff2e63", opacity=0.6, textposition='outside')
fig.show()
fig = px.bar(flg.sort_values('Recovered', ascending=False).head(20).sort_values('Recovered', ascending=True), 
             x="Recovered", y="Country/Region", title='Recovered', text='Recovered', orientation='h', 
             width=700, height=700, range_x = [0, max(flg['Recovered'])+10000])
fig.update_traces(marker_color='#21bf73', opacity=0.6, textposition='outside')
fig.show()
fig = px.bar(flg.sort_values('Active', ascending=False).head(20).sort_values('Active', ascending=True), 
             x="Active", y="Country/Region", title='Active', text='Active', orientation='h', 
             width=700, height=700, range_x = [0, max(flg['Active'])+3000])
fig.update_traces(marker_color='#fe9801', opacity=0.6, textposition='outside')
fig.show()
temp = covid_df.groupby(['Country/Region', 'Date'])['Confirmed', 'Deaths', 'Recovered'].sum()
temp = temp.reset_index()

fig = px.bar(temp, x="Date", y="Confirmed", color='Country/Region', orientation='v', height=600,
             title='Confirmed', color_discrete_sequence = px.colors.cyclical.mygbm)
fig.show()

fig = px.bar(temp, x="Date", y="Deaths", color='Country/Region', orientation='v', height=600,
             title='Deaths', color_discrete_sequence = px.colors.cyclical.mygbm)
fig.show()
temp = covid_df.groupby(['Country/Region', 'Date', ])['Confirmed', 'Deaths', 'Recovered']
temp = temp.sum().diff().reset_index()

mask = temp['Country/Region'] != temp['Country/Region'].shift(1)

temp.loc[mask, 'Confirmed'] = np.nan
temp.loc[mask, 'Deaths'] = np.nan
temp.loc[mask, 'Recovered'] = np.nan

fig = px.bar(temp, x="Date", y="Confirmed", color='Country/Region',title='New cases')
fig.show()
# html embedding
from IPython.display import Javascript
from IPython.core.display import display
from IPython.core.display import HTML
HTML('''<div class="flourish-embed flourish-bar-chart-race" data-src="visualisation/1571387"><script src="https://public.flourish.studio/resources/embed.js"></script></div>''')
province_state = pd.pivot_table(covid_df,index=["Province/State"] ,aggfunc=np.sum).sort_values(by='Confirmed', ascending=False)
province_state[:5].plot(kind='pie', subplots=True, figsize=(50, 50))
# Produce quick summary for China with total numbers
china_latest = covid_latest[covid_latest['Country/Region']=='China']

covid_ch =  covid_latest[(covid_latest['Country/Region'] == 'China')]
china_df = china_latest.groupby('Date')['Confirmed', 'Deaths', 'Recovered'].sum().reset_index()

summary_china = china_df.sort_values('Date', ascending=False)
summary_china.head(1).style.background_gradient(cmap='OrRd')
cl = china_latest.groupby('Province/State')['Confirmed', 'Deaths', 'Recovered'].sum()
cl = cl.reset_index().sort_values(by='Confirmed', ascending=False).reset_index(drop=True)
# cl.head().style.background_gradient(cmap='rainbow')

ncl = cl.copy()
ncl['Affected'] = ncl['Confirmed'] - ncl['Deaths'] - ncl['Recovered']
ncl = ncl.melt(id_vars="Province/State", value_vars=['Affected', 'Recovered', 'Deaths'])

fig = px.bar(ncl.sort_values(['variable', 'value']), 
             y="Province/State", x="value", color='variable', orientation='h', height=800,
             title='Number of Cases in China', color_discrete_sequence=["#ff2e63", '#21bf73', '#fe9801'])
fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
fig.show()
India = covid_df[covid_df['Country/Region']=='India']
Italy = covid_df[covid_df['Country/Region']=='Italy']
row = covid_df[covid_df['Country/Region']!='India']

India_latest = covid_latest[covid_latest['Country/Region'] == 'India']
Italy_latest = covid_latest[covid_latest['Country/Region'] == 'Italy']
row_latest = covid_latest[covid_latest['Country/Region']!='India']

India_latest_grouped = India_latest.groupby('Province/State')['Confirmed', 'Deaths', 'Recovered'].sum().reset_index()
Italy_latest_grouped = Italy_latest.groupby('Province/State')['Confirmed', 'Deaths', 'Recovered'].sum().reset_index()
row_latest_grouped = row_latest.groupby('Country/Region')['Confirmed', 'Deaths', 'Recovered'].sum().reset_index()
Italy_latest.head()
India_latest_grouped.sort_values(by='Deaths', ascending=False).head().style.background_gradient(cmap='Pastel1_r')
plot_india_over_time = covid_df[(covid_df['Country/Region']=='India') & (covid_df['Confirmed']!=0)]
plot_italy_over_time = covid_df[(covid_df['Country/Region']=='Italy') & (covid_df['Confirmed']!=0)]
plot_india_over_time['day'] = pd.to_datetime(plot_india_over_time['Date'], format='%Y-%m-%d')
plot_italy_over_time['day'] = pd.to_datetime(plot_italy_over_time['Date'], format='%Y-%m-%d')
from matplotlib.dates import DateFormatter

# Create figure and plot space
fig, ax = plt.subplots(figsize=(12, 12))

# Add x-axis and y-axis
ax.bar(plot_india_over_time['day'],
       plot_india_over_time['Confirmed'],
       color='purple')

# Set title and labels for axes
ax.set(xlabel="Date",
       ylabel="Confirmed",
       title="Number of cases confirmed over time in India")

# Define the date format
date_form = DateFormatter("%m-%d")
ax.xaxis.set_major_formatter(date_form)

plt.show()
fig, bx = plt.subplots(figsize=(12, 12))

# Add x-axis and y-axis
bx.bar(plot_italy_over_time['day'],
       plot_italy_over_time['Confirmed'],
       color='cyan')

# Set title and labels for axes
bx.set(xlabel="Date",
       ylabel="Confirmed",
       title="Number of cases confirmed over time in Italy")

# Define the date format
date_form = DateFormatter("%m-%d")
bx.xaxis.set_major_formatter(date_form)

plt.show()
# India 
m = folium.Map(location=[20.5937, 78.9629], tiles='cartodbpositron',
               min_zoom=3, max_zoom=6, zoom_start=5)

for i in range(0, len(India_latest)):
    folium.Circle(
        location=[India_latest.iloc[i]['Lat'], India_latest.iloc[i]['Long']],
        color='crimson', 
        tooltip =   '<li><bold>Country : '+str(India_latest.iloc[i]['Country/Region'])+
                    '<li><bold>Province : '+str(India_latest.iloc[i]['Province/State'])+
                    '<li><bold>Confirmed : '+str(India_latest.iloc[i]['Confirmed'])+
                    '<li><bold>Deaths : '+str(India_latest.iloc[i]['Deaths'])+
                    '<li><bold>Recovered : '+str(India_latest.iloc[i]['Recovered']),
        radius=int(India_latest.iloc[i]['Confirmed'])**1).add_to(m)
m
# India 
m = folium.Map(location=[41.8719, 12.5674], tiles='cartodbpositron',
               min_zoom=3, max_zoom=6, zoom_start=5)

for i in range(0, len(Italy_latest)):
    folium.Circle(
        location=[Italy_latest.iloc[i]['Lat'], Italy_latest.iloc[i]['Long']],
        color='crimson', 
        tooltip =   '<li><bold>Country : '+str(Italy_latest.iloc[i]['Country/Region'])+
                    '<li><bold>Province : '+str(Italy_latest.iloc[i]['Province/State'])+
                    '<li><bold>Confirmed : '+str(Italy_latest.iloc[i]['Confirmed'])+
                    '<li><bold>Deaths : '+str(Italy_latest.iloc[i]['Deaths'])+
                    '<li><bold>Recovered : '+str(Italy_latest.iloc[i]['Recovered']),
        radius=int(Italy_latest.iloc[i]['Confirmed'])**1).add_to(m)
m
covid_df.loc[covid_df['Country/Region']=='India']
for k,v in covid_df.groupby(['Country/Region','Province/State']):    
    for d in range(5):        
        covid_df.loc[v.index, f'Confirmed_Lag_{d}'] = v['Confirmed'].shift(d+1)
India_df = covid_df.fillna(0)
X_tr = [c for c in India_df.columns if 'Lag_' in c]
India_df[X_tr]
from lightgbm import LGBMRegressor    
model = LGBMRegressor()
model.fit(X=India_df[X_tr], y=India_df['Confirmed'])
from datetime import timedelta
pred_steps = 23

history = India_df.loc[India_df['Country/Region']=='India']
history0 = history.iloc[-1]
pred_init = history0[X_tr].values
pred_init_confirmed = history0['Confirmed']
# Test out of sample input
print('History 0: ', pred_init)
pred_init = np.roll(pred_init, 1)
pred_init[0] = pred_init_confirmed
print('Pred 0: ', pred_init)

pred = np.zeros(pred_steps)
for d in range(pred_steps):
    y = model.predict(pred_init.reshape(1,-1))
    pred_init = np.roll(pred_init, 1)
    pred_init[0] = y    
    pred[d] = y
    
dt_rng = pd.date_range(start=history0['Date']+timedelta(days=1), end=history0['Date']+timedelta(days=pred_steps),freq='D').values
preds = pd.Series(data=pred, index=dt_rng, )
from matplotlib import pyplot as plt
fig, ax = plt.subplots(figsize=(16,5))

history.set_index(['Date'])['Confirmed'].plot(ax=ax, marker='o')
preds.plot(ax=ax, marker='o')

plt.tight_layout()
covid_df.loc[covid_df['Country/Region']=='Italy']
for k,v in covid_df.groupby(['Country/Region','Province/State']):    
    for d in range(5):        
        covid_df.loc[v.index, f'Confirmed_Lag_{d}'] = v['Confirmed'].shift(d+1)
Italy_df = covid_df.fillna(0)
X_tr = [c for c in India_df.columns if 'Lag_' in c]
Italy_df[X_tr]
from lightgbm import LGBMRegressor    
model = LGBMRegressor()
model.fit(X=Italy_df[X_tr], y=Italy_df['Confirmed'])
from datetime import timedelta
pred_steps = 23

history = Italy_df.loc[Italy_df['Country/Region']=='Italy']
history0 = history.iloc[-1]
pred_init = history0[X_tr].values
pred_init_confirmed = history0['Confirmed']
# Test out of sample input
print('History 0: ', pred_init)
pred_init = np.roll(pred_init, 1)
pred_init[0] = pred_init_confirmed
print('Pred 0: ', pred_init)

pred = np.zeros(pred_steps)
for d in range(pred_steps):
    y = model.predict(pred_init.reshape(1,-1))
    pred_init = np.roll(pred_init, 1)
    pred_init[0] = y    
    pred[d] = y
    
dt_rng = pd.date_range(start=history0['Date']+timedelta(days=1), end=history0['Date']+timedelta(days=pred_steps),freq='D').values
preds = pd.Series(data=pred, index=dt_rng, )
from matplotlib import pyplot as plt
fig, ax = plt.subplots(figsize=(16,5))

history.set_index(['Date'])['Confirmed'].plot(ax=ax, marker='o')
preds.plot(ax=ax, marker='o')

plt.tight_layout()
