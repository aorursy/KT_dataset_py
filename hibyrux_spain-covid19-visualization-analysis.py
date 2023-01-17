! pip install calmap
!python --version
# storing and anaysis
import numpy as np
import pandas as pd

# visualization
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

from plotnine import *
import calmap

import plotly.express as px
import folium

# color pallette

c = '#393e46'
d = '#ff2e63'
r = '#30e3ca'
i = '#f8b400'
cdr = [c, d, r] # grey - red - blue
idr = [i, d, r] # yellow - red - blue
!ls ../input/covid19
# importing datasets
full_table = pd.read_csv('../input/covid19/complete_data.csv',parse_dates=['Date'])
full_table.head()
# dataframe info
full_table.info()
# checking for missing value
# full_table.isna().sum()
# replacing Mainland china with just China
full_table['Country/Region'] = full_table['Country/Region'].replace('Mainland China', 'China')

# filling missing values with NA
full_table[['Province/State']] = full_table[['Province/State']].fillna('NA')
full_table[['Confirmed', 'Deaths']] = full_table[['Confirmed', 'Deaths']].fillna(0)
# complete dataset 
# complete = full_table.copy()

# full table
spain = full_table[full_table['Country/Region']=='Spain']
row = full_table[full_table['Country/Region']!='Spain']

# latest
full_latest = full_table[full_table['Date'] == max(full_table['Date'])].reset_index()
spain_latest = full_latest[full_latest['Country/Region']=='Spain']
row_latest = full_latest[full_latest['Country/Region']!='Spain']

# latest condensed
full_latest_grouped = full_latest.groupby('Country/Region')['Confirmed', 'Deaths'].sum().reset_index()
spain_latest_grouped = spain_latest.groupby('Province/State')['Confirmed', 'Deaths'].sum().reset_index()
row_latest_grouped = row_latest.groupby('Country/Region')['Confirmed', 'Deaths'].sum().reset_index()
temp = full_table.groupby(['Country/Region', 'Province/State'])['Confirmed', 'Deaths'].max()
temp.style.background_gradient(cmap='Pastel1_r')
temp = full_table.groupby('Date')['Confirmed', 'Deaths'].sum()
temp = temp.reset_index()
temp = temp.sort_values('Date', ascending=False)
temp.head(1).reset_index(drop=True).style.background_gradient(cmap='Pastel1')
temp_f = full_latest_grouped[['Country/Region', 'Confirmed', 'Deaths']]
temp_f = temp_f.sort_values(by='Confirmed', ascending=False)
temp_f = temp_f.reset_index(drop=True)
temp_f.style.background_gradient(cmap='Pastel1_r')
temp_flg = full_latest_grouped[['Country/Region', 'Deaths']]
temp_flg = temp_flg.sort_values(by='Deaths', ascending=False)
temp_flg = temp_flg.reset_index(drop=True)
temp_flg = temp_flg[temp_flg['Deaths']>0]
temp_flg.style.background_gradient(cmap='Reds')
temp = row_latest_grouped[row_latest_grouped['Confirmed']==
                          row_latest_grouped['Deaths']]
temp = temp[['Country/Region', 'Confirmed', 'Deaths']]
temp = temp.sort_values('Confirmed', ascending=False)
temp = temp.reset_index(drop=True)
temp.style.background_gradient(cmap='Reds')
temp = row_latest_grouped[row_latest_grouped['Confirmed']==
                          row_latest_grouped['Deaths']]
temp = temp[['Country/Region', 'Confirmed', 'Deaths']]
temp = temp.sort_values('Confirmed', ascending=False)
temp = temp.reset_index(drop=True)
temp.style.background_gradient(cmap='Greens')
temp_f = spain_latest_grouped[['Province/State', 'Confirmed', 'Deaths']]
temp_f = temp_f.sort_values(by='Confirmed', ascending=False)
temp_f = temp_f.reset_index(drop=True)
temp_f.style.background_gradient(cmap='Pastel1_r')
temp = spain_latest_grouped[spain_latest_grouped['Confirmed']==
                          spain_latest_grouped['Deaths']]
temp = temp[['Province/State', 'Confirmed', 'Deaths']]
temp = temp.sort_values('Confirmed', ascending=False)
temp = temp.reset_index(drop=True)
temp.style.background_gradient(cmap='Greens')
# World wide

m = folium.Map(location=[0, 0], tiles='cartodbpositron',
               min_zoom=1, max_zoom=4, zoom_start=1)

for i in range(0, len(full_latest)):
    folium.Circle(
        location=[full_latest.iloc[i]['Lat'], full_latest.iloc[i]['Long']],
        color='crimson', 
        tooltip =   '<li><bold>Country : '+str(full_latest.iloc[i]['Country/Region'])+
                    '<li><bold>Province : '+str(full_latest.iloc[i]['Province/State'])+
                    '<li><bold>Confirmed : '+str(full_latest.iloc[i]['Confirmed'])+
                    '<li><bold>Deaths : '+str(full_latest.iloc[i]['Deaths']),
        radius=int(full_latest.iloc[i]['Confirmed'])).add_to(m)
m
# Spain 

m = folium.Map(location=[30, 116], tiles='cartodbpositron',
               min_zoom=2, max_zoom=5, zoom_start=3)

for i in range(0, len(spain_latest)):
    folium.Circle(
        location=[spain_latest.iloc[i]['Lat'], spain_latest.iloc[i]['Long']],
        color='crimson', 
        tooltip =   '<li><bold>Country : '+str(spain_latest.iloc[i]['Country/Region'])+
                    '<li><bold>Province : '+str(spain_latest.iloc[i]['Province/State'])+
                    '<li><bold>Confirmed : '+str(spain_latest.iloc[i]['Confirmed'])+
                    '<li><bold>Deaths : '+str(spain_latest.iloc[i]['Deaths']),
        radius=int(spain_latest.iloc[i]['Confirmed'])**1).add_to(m)
m
fig = px.choropleth(full_latest_grouped, locations="Country/Region", 
                    locationmode='country names', color="Confirmed", 
                    hover_name="Country/Region", range_color=[1,2000], 
                    color_continuous_scale="aggrnyl", 
                    title='Countries with Confirmed Cases')
fig.update(layout_coloraxis_showscale=False)
fig.show()

# ------------------------------------------------------------------------

fig = px.choropleth(full_latest_grouped[full_latest_grouped['Deaths']>0], 
                    locations="Country/Region", locationmode='country names',
                    color="Deaths", hover_name="Country/Region", 
                    range_color=[1,50], color_continuous_scale="agsunset",
                    title='Countries with Deaths Reported')
fig.update(layout_coloraxis_showscale=False)
fig.show()
formated_gdf = full_table.groupby(['Date', 'Country/Region'])['Confirmed', 'Deaths'].max()
formated_gdf = formated_gdf.reset_index()
formated_gdf = formated_gdf[formated_gdf['Country/Region']!='China']
formated_gdf['Date'] = pd.to_datetime(formated_gdf['Date'])
formated_gdf['Date'] = formated_gdf['Date'].dt.strftime('%m/%d/%Y')
formated_gdf['size'] = formated_gdf['Confirmed'].pow(0.5)

fig = px.scatter_geo(formated_gdf[formated_gdf['Country/Region']!='China'], 
                     locations="Country/Region", locationmode='country names', 
                     color="Confirmed", size='size', hover_name="Country/Region", 
                     range_color= [0, max(formated_gdf['Confirmed'])+2], 
                     projection="natural earth", animation_frame="Date", 
                     title='Spread outside China over time')
fig.update(layout_coloraxis_showscale=False)
fig.show()
formated_gdf = full_table.groupby(['Date', 'Country/Region'])['Confirmed', 'Deaths'].max()
formated_gdf = formated_gdf.reset_index()
formated_gdf = formated_gdf[formated_gdf['Country/Region']!='Spain']
formated_gdf['Date'] = pd.to_datetime(formated_gdf['Date'])
formated_gdf['Date'] = formated_gdf['Date'].dt.strftime('%m/%d/%Y')
formated_gdf['size'] = formated_gdf['Confirmed'].pow(0.5)

fig = px.scatter_geo(formated_gdf[formated_gdf['Country/Region']!='Spain'], 
                     locations="Country/Region", locationmode='country names', 
                     color="Confirmed", size='size', hover_name="Country/Region", 
                     range_color= [0, max(formated_gdf['Confirmed'])+2], 
                     projection="natural earth", animation_frame="Date", 
                     title='Spread outside Spain over time')
fig.update(layout_coloraxis_showscale=False)
fig.show()
fig = px.bar(full_latest_grouped[['Country/Region', 'Confirmed']].sort_values('Confirmed', ascending=False), 
             x="Confirmed", y="Country/Region", color='Country/Region', orientation='h',
             log_x=True, color_discrete_sequence = px.colors.qualitative.Bold, title='Confirmed Cases', width=900, height=1200)
fig.show()

temp = full_latest_grouped[['Country/Region', 'Deaths']].sort_values('Deaths', ascending=False)
fig = px.bar(temp[temp['Deaths']>0], 
             x="Deaths", y="Country/Region", color='Country/Region', title='Deaths', orientation='h',
             log_x=True, color_discrete_sequence = px.colors.qualitative.Bold, width=900)
fig.show()
temp = full_table.groupby(['Country/Region', 'Date'])['Confirmed', 'Deaths'].sum()
temp = temp.reset_index()
# temp.head()

fig = px.bar(temp, x="Date", y="Confirmed", color='Country/Region', orientation='v', height=600,
             title='Confirmed', color_discrete_sequence = px.colors.cyclical.mygbm)
fig.show()

fig = px.bar(temp, x="Date", y="Deaths", color='Country/Region', orientation='v', height=600,
             title='Deaths', color_discrete_sequence = px.colors.cyclical.mygbm)
fig.show()
# In Spain
temp = spain.groupby('Date')['Confirmed', 'Deaths'].sum().diff()
temp = temp.reset_index()
temp = temp.melt(id_vars="Date", 
                 value_vars=['Confirmed', 'Deaths'])

fig = px.bar(temp, x="Date", y="value", color='variable', 
             title='In Spain',
             color_discrete_sequence=cdr)
fig.update_layout(barmode='group')
fig.show()

#-----------------------------------------------------------------------------

# ROW
temp = row.groupby('Date')['Confirmed', 'Deaths'].sum().diff()
temp = temp.reset_index()
temp = temp.melt(id_vars="Date", 
                 value_vars=['Confirmed', 'Deaths'])

fig = px.bar(temp, x="Date", y="value", color='variable', 
             title='Outside Spain',
             color_discrete_sequence=cdr)
fig.update_layout(barmode='group')
fig.show()
def from_spain_or_not(row):
    if row['Country/Region']=='Spain':
        return 'From Spain'
    else:
        return 'Outside Spain'
    
temp = full_table.copy()
temp['Region'] = temp.apply(from_spain_or_not, axis=1)
temp = temp.groupby(['Region', 'Date'])['Confirmed', 'Deaths']
temp = temp.sum().diff().reset_index()
mask = temp['Region'] != temp['Region'].shift(1)
temp.loc[mask, 'Confirmed'] = np.nan
temp.loc[mask, 'Deaths'] = np.nan

fig = px.bar(temp, x='Date', y='Confirmed', color='Region', barmode='group', 
             text='Confirmed', title='Confirmed', color_discrete_sequence= cdr)
fig.update_traces(textposition='outside')
fig.show()

fig = px.bar(temp, x='Date', y='Deaths', color='Region', barmode='group', 
             text='Confirmed', title='Deaths', color_discrete_sequence= cdr)
fig.update_traces(textposition='outside')
fig.update_traces(textangle=-90)
fig.show()
temp = full_table.groupby(['Country/Region', 'Date', ])['Confirmed', 'Deaths']
temp = temp.sum().diff().reset_index()

mask = temp['Country/Region'] != temp['Country/Region'].shift(1)

temp.loc[mask, 'Confirmed'] = np.nan
temp.loc[mask, 'Deaths'] = np.nan

fig = px.bar(temp, x="Date", y="Confirmed", color='Country/Region',
             title='Number of new cases everyday')
fig.show()

fig = px.bar(temp[temp['Country/Region']!='Spain'], x="Date", y="Confirmed", color='Country/Region',
             title='Number of new cases outside Spain everyday')
fig.show()

fig = px.bar(temp, x="Date", y="Deaths", color='Country/Region',
             title='Number of new death case reported outside Spain everyday')
fig.show()

fig = px.bar(temp[temp['Country/Region']!='Spain'], x="Date", y="Deaths", color='Country/Region',
             title='Number of new death case reported outside Spain everyday')
fig.show()
c_spread = spain[spain['Confirmed']!=0].groupby('Date')['Province/State'].unique().apply(len)
c_spread = pd.DataFrame(c_spread).reset_index()

fig = px.line(c_spread, x='Date', y='Province/State', 
              title='Number of Provinces/States/Regions of Spain to which COVID-19 spread over the time',
             color_discrete_sequence=cdr)
fig.show()

# ------------------------------------------------------------------------------------------

spread = full_table[full_table['Confirmed']!=0].groupby('Date')['Country/Region'].unique().apply(len)
spread = pd.DataFrame(spread).reset_index()

fig = px.line(spread, x='Date', y='Country/Region', 
              title='Number of Countries/Regions to which COVID-19 spread over the time',
             color_discrete_sequence=cdr)
fig.show()
rl = row_latest.groupby('Country/Region')['Confirmed', 'Deaths'].sum()
rl = rl.reset_index().sort_values(by='Confirmed', ascending=False).reset_index(drop=True)
rl.head().style.background_gradient(cmap='rainbow')

ncl = rl.copy()
ncl['Affected'] = ncl['Confirmed'] - ncl['Deaths']
ncl = ncl.melt(id_vars="Country/Region", value_vars=['Affected', 'Deaths'])

fig = px.bar(ncl.sort_values(['variable', 'value']), 
             x="Country/Region", y="value", color='variable', orientation='v', height=800,
             # height=600, width=1000,
             title='Number of Cases outside Spain', color_discrete_sequence=cdr)
fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
fig.show()

# ------------------------------------------

cl = spain_latest.groupby('Province/State')['Confirmed', 'Deaths'].sum()
cl = cl.reset_index().sort_values(by='Confirmed', ascending=False).reset_index(drop=True)
# cl.head().style.background_gradient(cmap='rainbow')

ncl = cl.copy()
ncl['Affected'] = ncl['Confirmed'] - ncl['Deaths']
ncl = ncl.melt(id_vars="Province/State", value_vars=['Affected','Deaths'])

fig = px.bar(ncl.sort_values(['variable', 'value']), 
             y="Province/State", x="value", color='variable', orientation='h', height=800,
             # height=600, width=1000,
             title='Number of Cases in Spain', color_discrete_sequence=cdr)
fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
fig.show()
fig = px.treemap(spain_latest.sort_values(by='Confirmed', ascending=False).reset_index(drop=True), 
                 path=["Province/State"], values="Confirmed",
                 title='Number of Confirmed Cases in Spain Provinces',
                 color_discrete_sequence = px.colors.qualitative.Prism)
fig.show()

fig = px.treemap(spain_latest.sort_values(by='Deaths', ascending=False).reset_index(drop=True), 
                 path=["Province/State"], values="Deaths", 
                 title='Number of Deaths Reported in Spain Provinces',
                 color_discrete_sequence = px.colors.qualitative.Prism)
fig.show()

# ----------------------------------------------------------------------------

fig = px.treemap(row_latest, path=["Country/Region"], values="Confirmed", 
                 title='Number of Confirmed Cases outside Spain',
                 color_discrete_sequence = px.colors.qualitative.Pastel)
fig.show()

fig = px.treemap(row_latest, path=["Country/Region"], values="Deaths", 
                 title='Number of Deaths outside Spain',
                 color_discrete_sequence = px.colors.qualitative.Pastel)
fig.show()

temp = full_table.groupby(['Date', 'Country/Region'])['Confirmed'].sum()
temp = temp.reset_index().sort_values(by=['Date', 'Country/Region'])
temp.head()
px.line(temp, x='Date', y='Confirmed')
temp = full_table.groupby(['Date', 'Country/Region'])['Confirmed'].sum()
temp = temp.reset_index().sort_values(by=['Date', 'Country/Region'])

plt.style.use('seaborn')
g = sns.FacetGrid(temp, col="Country/Region", hue="Country/Region", 
                  sharey=False, col_wrap=5)
g = g.map(plt.plot, "Date", "Confirmed")
g.set_xticklabels(rotation=90)
plt.show()
temp = full_table.groupby(['Country/Region', 'Date', ])['Confirmed', 'Deaths']
temp = temp.sum().diff().reset_index()

mask = temp['Country/Region'] != temp['Country/Region'].shift(1)

temp.loc[mask, 'Confirmed'] = np.nan
temp.loc[mask, 'Deaths'] = np.nan

plt.style.use('seaborn')
g = sns.FacetGrid(temp, col="Country/Region", hue="Country/Region", 
                  sharey=False, col_wrap=5)
g = g.map(sns.lineplot, "Date", "Confirmed")
g.set_xticklabels(rotation=90)
plt.show()
temp = full_table.groupby('Date')['Confirmed'].sum()
temp = temp.diff()

plt.figure(figsize=(20, 5))
calmap.yearplot(temp, fillcolor='white', cmap='Reds', linewidth=0.5)
plt.plot()
spread = full_table[full_table['Confirmed']!=0].groupby('Date')
spread = spread['Country/Region'].unique().apply(len).diff()

plt.figure(figsize=(20, 5))
calmap.yearplot(spread, fillcolor='white', cmap='Greens', linewidth=0.5)
plt.plot()