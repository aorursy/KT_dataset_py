# storing and anaysis
import numpy as np
import pandas as pd

# visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import folium
from matplotlib.dates import DateFormatter
!ls -lt ../input/corona-virus-report
# # importing datasets

full_table = pd.read_csv('../input/corona-virus-report/covid_19_clean_complete.csv', 
                         parse_dates=['Date'])
full_table.sort_values('Recovered', ascending=False).head()
# converting to proper date format
full_table['Date'] = pd.to_datetime(full_table['Date'])

# replacing Mainland china with just China
full_table['Country/Region'] = full_table['Country/Region'].replace('Mainland China', 'China')

# filling missing values with 0 in columns ('Confirmed', 'Deaths', 'Recovered')
full_table[['Confirmed', 'Deaths', 'Recovered']] = full_table[['Confirmed', 'Deaths', 'Recovered']].fillna(0)
# adding two more columns
full_table['Deaths to 1000 Confirmed'] = round(full_table['Deaths']/full_table['Confirmed'], 3)
full_table['Recovered to 1000 Confirmed'] = round(full_table['Recovered']/full_table['Confirmed'], 3)
full_table['Recovered to 1000 Deaths'] = round(full_table['Recovered']/full_table['Deaths'], 3)

# cases in the Diamond Princess cruise ship
ship = full_table[full_table['Province/State']=='Diamond Princess cruise ship']

########
full_table = full_table[full_table['Province/State']!='Diamond Princess cruise ship']
full_table.sort_values('Deaths', ascending=False).head()
full_table.describe()
china = full_table[full_table['Country/Region']=='China']

India = full_table[full_table['Country/Region']=='India']

Italy = full_table[full_table['Country/Region']=='Italy']
row = full_table[full_table['Country/Region']!='India']

full_latest = full_table[full_table['Date'] == max(full_table['Date'])].reset_index()
china_latest = full_latest[full_latest['Country/Region']=='China']
India_latest = full_latest[full_latest['Country/Region'] == 'India']
Italy_latest = full_latest[full_latest['Country/Region'] == 'Italy']
row_latest = full_latest[full_latest['Country/Region']!='India']

full_latest_grouped = full_latest.groupby('Country/Region')['Confirmed', 'Deaths', 'Recovered'].sum().reset_index()
china_latest_grouped = china_latest.groupby('Province/State')['Confirmed', 'Deaths', 'Recovered'].sum().reset_index()
India_latest_grouped = India_latest.groupby('Province/State')['Confirmed', 'Deaths', 'Recovered'].sum().reset_index()
Italy_latest_grouped = Italy_latest.groupby('Province/State')['Confirmed', 'Deaths', 'Recovered'].sum().reset_index()
row_latest_grouped = row_latest.groupby('Country/Region')['Confirmed', 'Deaths', 'Recovered'].sum().reset_index()
Italy_latest.head()
china_latest_grouped.sort_values(by='Deaths', ascending=False).head().style.background_gradient(cmap='Pastel1_r')
India_latest_grouped.sort_values(by='Deaths', ascending=False).head().style.background_gradient(cmap='Pastel1_r')
temp_f = full_latest_grouped[['Country/Region', 'Confirmed']].sort_values(by='Confirmed', ascending=False).reset_index(drop=True)
temp_f.head(10).style.background_gradient(cmap='Pastel1_r')
temp_c = china_latest_grouped[['Province/State', 'Confirmed']].sort_values(by='Confirmed', ascending=False).reset_index(drop=True)
temp_c.head(10).style.background_gradient(cmap='Pastel1_r')
temp_flg = full_latest_grouped[['Country/Region', 'Deaths']].sort_values(by='Deaths', ascending=False).reset_index(drop=True)
temp_flg = temp_flg[temp_flg['Deaths']>0]
temp_flg.style.background_gradient(cmap='Pastel1_r')
temp = full_table.groupby('Date')['Confirmed', 'Deaths', 'Recovered'].sum().reset_index().sort_values('Date', ascending=False)
temp.head(1).style.background_gradient(cmap='Pastel1')
# Cases in the Diamond Princess cruise ship
ship.sort_values(by='Date', ascending=False).head(1)[['Confirmed', 'Deaths', 'Recovered']].style.background_gradient(cmap='Pastel2')
# Number of Countries/Regions to which COVID-19 spread

print(len(temp_f))
# Number of Province/State in Mainland China to which COVID-19 spread

len(temp_c)
# Number of countries with deaths reported

len(temp_flg)
full_table = full_table.fillna(0)

full_table.sort_values('Deaths', ascending= False).head()
# temp1 = full_table.groupby("Country/Region")['Date','Confirmed','Deaths','Recovered'].sum().reset_index(drop=True).sort_values('Date', ascending=False).head(1).style.background_gradient(cmap='Pastel1')
plot_india_over_time = full_table[(full_table['Country/Region']=='India') & (full_table['Confirmed']!=0)]

plot_italy_over_time = full_table[(full_table['Country/Region']=='Italy') & (full_table['Confirmed']!=0)]
plot_india_over_time['day'] = pd.to_datetime(plot_india_over_time['Date'], format='%Y-%m-%d')
plot_italy_over_time['day'] = pd.to_datetime(plot_italy_over_time['Date'], format='%Y-%m-%d')
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
       color='purple')

# Set title and labels for axes
bx.set(xlabel="Date",
       ylabel="Confirmed",
       title="Number of cases confirmed over time in Italy")

# Define the date format
date_form = DateFormatter("%m-%d")
bx.xaxis.set_major_formatter(date_form)

plt.show()
# World wide

m = folium.Map(location=[0, 0], tiles='cartodbpositron',
               min_zoom=2, max_zoom=4, zoom_start=2)

for i in range(0, len(full_latest)):
    folium.Circle(
        location=[full_latest.iloc[i]['Lat'], full_latest.iloc[i]['Long']],
        color='crimson', 
        tooltip =   '<li><bold>Country : '+str(full_latest.iloc[i]['Country/Region'])+
                    '<li><bold>Province : '+str(full_latest.iloc[i]['Province/State'])+
                    '<li><bold>Confirmed : '+str(full_latest.iloc[i]['Confirmed'])+
                    '<li><bold>Deaths : '+str(full_latest.iloc[i]['Deaths'])+
                    '<li><bold>Recovered : '+str(full_latest.iloc[i]['Recovered']),
        radius=int(full_latest.iloc[i]['Confirmed'])).add_to(m)
m
# India 
m = folium.Map(location=[20.59, 78.96], tiles='cartodbpositron',
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
# Italy 
# m = folium.Map(location=[41.87, 12.56], tiles='cartodbpositron',
#                min_zoom=3, max_zoom=6, zoom_start=5)

# for i in range(0, len(Italy_latest)):
#     folium.Circle(
#         location=[Italy_latest.iloc[i]['Lat'], Italy_latest.iloc[i]['Long']],
#         color='crimson', 
#         tooltip =   '<li><bold>Country : '+str(Italy_latest.iloc[i]['Country/Region'])+
#                     '<li><bold>Province : '+str(Italy_latest.iloc[i]['Province/State'])+
#                     '<li><bold>Confirmed : '+str(Italy_latest.iloc[i]['Confirmed'])+
#                     '<li><bold>Deaths : '+str(Italy_latest.iloc[i]['Deaths'])+
#                     '<li><bold>Recovered : '+str(Italy_latest.iloc[i]['Recovered']),
#         radius=int(Italy_latest.iloc[i]['Confirmed'])**1).add_to(m)
# px.choropleth(full_latest_grouped, locations='Country/Region', color='Confirmed',color_continuous_scale="Viridis")

fig = px.choropleth(full_latest_grouped, locations="Country/Region", locationmode='country names', 
                    color="Confirmed", hover_name="Country/Region", range_color=[1,50], color_continuous_scale="Sunsetdark", 
                    title='Countries with Confirmed Cases')
fig.update(layout_coloraxis_showscale=False)
fig.show()

fig = px.choropleth(full_latest_grouped[full_latest_grouped['Deaths']>0], locations="Country/Region", locationmode='country names',
                    color="Deaths", hover_name="Country/Region", range_color=[1,50], color_continuous_scale="Peach",
                    title='Countries with Deaths Reported')
fig.update(layout_coloraxis_showscale=False)
fig.show()
gdf = full_table.groupby(['Date', 'Country/Region'])['Confirmed', 'Deaths', 'Recovered'].max().reset_index()
formated_gdf = gdf.copy()
formated_gdf = formated_gdf[formated_gdf['Country/Region']!='India']
formated_gdf['Date'] = pd.to_datetime(formated_gdf['Date'])
formated_gdf['Date'] = formated_gdf['Date'].dt.strftime('%m/%d/%Y')

fig = px.scatter_geo(formated_gdf[formated_gdf['Country/Region']!='India'], locations="Country/Region", locationmode='country names', 
                     color="Confirmed", size='Confirmed', hover_name="Country/Region", range_color= [0, max(formated_gdf['Confirmed'])+2], 
                     projection="natural earth", animation_frame="Date", title='Spread outside India over time')
fig.update(layout_coloraxis_showscale=False)
fig.show()

# -----------------------------------------------------------------------------------

India_map = India.groupby(['Date'])['Confirmed', 'Deaths', 'Recovered', 'Lat', 'Long'].max().reset_index()
India_map['size'] = India_map['Confirmed'].pow(0.5)
India_map['Date'] = pd.to_datetime(India_map['Date'])
India_map['Date'] = India_map['Date'].dt.strftime('%m/%d/%Y')
India_map.head()

fig = px.scatter_geo(India_map, lat='Lat', lon='Long', scope='asia',
                     color="size", size='size', hover_name='Confirmed', hover_data=['Confirmed', 'Deaths', 'Recovered'],
                     projection="natural earth", animation_frame="Date", title='Spread in India over time')
fig.update(layout_coloraxis_showscale=False)
fig.show()
# c_spread.head()
c_spread = pd.DataFrame(china[china['Confirmed']!=0].groupby('Date')['Province/State'].unique().apply(len)).reset_index()
fig = px.line(c_spread, x='Date', y='Province/State', title='Number of Provinces/States/Regions of China to which COVID-19 spread over the time')
fig.show()

spread = pd.DataFrame(full_table[full_table['Confirmed']!=0].groupby('Date')['Country/Region'].unique().apply(len)).reset_index()
fig = px.line(spread, x='Date', y='Country/Region', title='Number of Countries/Regions to which COVID-19 spread over the time')
fig.show()
rl = row_latest.groupby('Country/Region')['Confirmed', 'Deaths', 'Recovered'].sum().reset_index().sort_values(by='Confirmed', ascending=False).reset_index(drop=True)
rl.head().style.background_gradient(cmap='rainbow')

ncl = rl.copy()
ncl['Affected'] = ncl['Confirmed'] - ncl['Deaths'] - ncl['Recovered']
ncl = ncl.melt(id_vars="Country/Region", value_vars=['Affected', 'Recovered', 'Deaths'])

fig = px.bar(ncl.sort_values(['variable', 'value']), 
             y="Country/Region", x="value", color='variable', orientation='h', height=800,
             # height=600, width=1000,
             title='Number of Cases outside China')
fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
fig.show()

# ------------------------------------------

cl = china_latest.groupby('Province/State')['Confirmed', 'Deaths', 'Recovered'].sum().reset_index().sort_values(by='Confirmed', ascending=False).reset_index(drop=True)
# cl.head().style.background_gradient(cmap='rainbow')

ncl = cl.copy()
ncl['Affected'] = ncl['Confirmed'] - ncl['Deaths'] - ncl['Recovered']
ncl = ncl.melt(id_vars="Province/State", value_vars=['Affected', 'Recovered', 'Deaths'])

fig = px.bar(ncl.sort_values(['variable', 'value']), 
             y="Province/State", x="value", color='variable', orientation='h', height=800,
             # height=600, width=1000,
             title='Number of Cases in China')
fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
fig.show()
temp = gdf[gdf['Country/Region']=='China'].reset_index()
temp = temp.melt(id_vars='Date', value_vars=['Confirmed', 'Deaths', 'Recovered'])
fig = px.bar(temp, x="Date", y="value", color='variable', facet_col="variable")
fig.show()
temp = gdf[gdf['Country/Region']!='China'].groupby('Date').sum().reset_index()
temp = temp.melt(id_vars='Date', value_vars=['Confirmed', 'Deaths', 'Recovered'])
fig = px.bar(temp, x="Date", y="value", color='variable', facet_col="variable")
fig.show()
fig = px.treemap(china_latest.sort_values(by='Confirmed', ascending=False).reset_index(drop=True), 
           path=["Province/State"], values="Confirmed", title='Number of Confirmed Cases in Chinese Provinces')
fig.show()

fig = px.treemap(china_latest.sort_values(by='Deaths', ascending=False).reset_index(drop=True), 
           path=["Province/State"], values="Deaths", title='Number of Deaths Reported in Chinese Provinces')
fig.show()

fig = px.treemap(china_latest.sort_values(by='Recovered', ascending=False).reset_index(drop=True), 
           path=["Province/State"], values="Recovered", title='Number of Recovered Cases in Chinese Provinces')
fig.show()
fig = px.treemap(row_latest, path=["Country/Region"], values="Confirmed", title='Number of Confirmed Cases outside china')
fig.show()

fig = px.treemap(row_latest, path=["Country/Region"], values="Deaths", title='Number of Deaths outside china')
fig.show()

fig = px.treemap(row_latest, path=["Country/Region"], values="Recovered", title='Number of Recovered Cases outside china')
fig.show()
fig = px.bar(china_latest.sort_values(by='Deaths to 1000 Confirmed', ascending=False).reset_index(drop=True), 
       x="Province/State", y="Deaths to 1000 Confirmed", title='Deaths to Confirmed ratio')
fig.show()

fig = px.bar(china_latest.sort_values(by='Recovered to 1000 Confirmed', ascending=False).reset_index(drop=True), 
       x="Province/State", y="Recovered to 1000 Confirmed", title='Recovered to Confirmed ratio')
fig.show()
temp = row.groupby('Date')['Confirmed', 'Deaths', 'Recovered'].sum().reset_index()
fig = px.scatter(temp, x="Date", y="Confirmed", color="Confirmed",
                 size='Confirmed', hover_data=['Confirmed'])
fig.show()