import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from urllib.request import urlopen
import json
import glob
import os   
import plotly.offline as py   
df = pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv")
df.info()
df.rename(columns={'Country/Region':'Country'}, inplace=True)
df.rename(columns={'Province/State':'States'}, inplace=True)
df.rename(columns={'ObservationDate':'Date'}, inplace=True)
df.head()

cnf, dth, rec = '#393e46', '#ff2e63', '#21bf73'
temp = df.groupby('Date')['Deaths', 'Confirmed'].max().reset_index()
temp = temp.melt(id_vars="Date", value_vars=['Deaths', 'Confirmed'],
                 var_name='Case', value_name='Count')
temp.head()

fig = px.area(temp, x="Date", y="Count", color='Case', height=600,
             title='Confirmed vs Death cases on a log scale', color_discrete_sequence = [dth, cnf])
fig.update_layout(xaxis_rangeslider_visible=True, yaxis_type = "log")
fig.show()

temp = df.groupby('Date')['Recovered', 'Deaths'].max().reset_index()
temp = temp.melt(id_vars="Date", value_vars=['Recovered', 'Deaths'],
                 var_name='Case', value_name='Count')
temp.head()

fig = px.area(temp, x="Date", y="Count", color='Case', height=600,
             title='Recovered vs Death cases on a log scale', color_discrete_sequence = [dth, cnf])
fig.update_layout(xaxis_rangeslider_visible=True, yaxis_type = "log")
fig.show()
temp = df.groupby('Date')['Recovered', 'Deaths', 'Confirmed'].max().reset_index()
temp = temp.melt(id_vars="Date", value_vars=['Recovered', 'Deaths', 'Confirmed'],
                 var_name='Case', value_name='Count')
temp.head()

fig = px.area(temp, x="Date", y="Count", color='Case', height=600,
             title='Cases over time', color_discrete_sequence = [rec, dth, cnf])
fig.update_layout(xaxis_rangeslider_visible=True)
fig.show()
fig = px.bar(df, x="Date", y="Confirmed", color='Country', height=600,
             title='Confirmed Cases on a log scale', color_discrete_sequence = px.colors.qualitative.Vivid)
fig.update_traces(textposition='outside')
fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide', yaxis_type = 'log')
fig.show()
fig = px.bar(df, x="Date", y="Recovered", color='Country', height=600,
             title='Recovered Cases on a log scale', color_discrete_sequence = px.colors.cyclical.mygbm)
fig.update_layout(yaxis_type = "log")
fig.show()
fig = px.bar(df, x="Date", y="Deaths", color='Country', height=600,
             title='Death Cases on a log scale', color_discrete_sequence = px.colors.cyclical.mygbm)
fig.update_layout(yaxis_type = "log")
fig.show()
df_con = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv")
df_con.rename(columns={'Country/Region':'Country'}, inplace=True)
df_con.rename(columns={'Province/State':'States'}, inplace=True)
df_con.rename(columns={'ObservationDate':'Date'}, inplace=True)
df.head()
df_con = df_con[["States","Lat","Long","Country"]]
df_temp = df.copy()
df_latlong = pd.merge(df_temp, df_con, on=["Country", "States"])
import folium
temp = df_latlong[df_latlong['Date'] == max(df_latlong['Date'])]

m = folium.Map(location=[0, 0], tiles='cartodbpositron',
               min_zoom=1, max_zoom=4, zoom_start=1)

for i in range(0, len(temp)):
    folium.Circle(
        location=[temp.iloc[i]['Lat'], temp.iloc[i]['Long']],
        color='crimson', fill='crimson',
        tooltip =   '<li><bold>Country : '+str(temp.iloc[i]['Country'])+
                    '<li><bold>Province : '+str(temp.iloc[i]['States'])+
                    '<li><bold>Confirmed : '+str(temp.iloc[i]['Confirmed'])+
                    '<li><bold>Deaths : '+str(temp.iloc[i]['Deaths']),
        radius=int(temp.iloc[i]['Confirmed'])**0.5).add_to(m)
m
fig_map = px.density_mapbox(df_latlong,lat="Lat",lon="Long",hover_name="States",hover_data=["Confirmed","Deaths","Recovered"],animation_frame="Date",
                        color_continuous_scale="Portland",radius=7,zoom=0,height=500)

fig_map.update_layout(title='Confirmed, Deaths, Recovered cases globally (Time lapse)',
                  font=dict(family="Courier New, monospace", size=18,color="#7f7f7f"))

fig_map.update_layout(mapbox_style="open-street-map", mapbox_center_lon=0)

fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0})


fig_map.show()
py.init_notebook_mode(connected=True)

#GroupingBy the dataset for the map
formated_gdf = df.groupby(['Date', 'Country'])['Confirmed'].max()
formated_gdf = formated_gdf.reset_index()
formated_gdf['Date'] = pd.to_datetime(formated_gdf['Date'])
formated_gdf['Date'] = formated_gdf['Date'].dt.strftime('%m/%d/%Y')

formated_gdf['ConfirmedCases'] = formated_gdf.Confirmed + 1


#Plotting the figure


fig = px.choropleth(formated_gdf, locations="Country", locationmode='country names', 
                     color="ConfirmedCases", hover_name="Country",projection="mercator",
                     animation_frame="Date",width=700, height=800,
                     color_continuous_scale=px.colors.sequential.Viridis,
                     title='COVID-19 Confirmed Cases Across World')

#Showing the figure
fig.update(layout_coloraxis_showscale=True)
py.offline.iplot(fig)

#Creating the interactive map for recovered cases across the world
py.init_notebook_mode(connected=True)

#GroupingBy the dataset for the map
formated_gdf = df.groupby(['Date', 'Country'])['Recovered'].max()
formated_gdf = formated_gdf.reset_index()
formated_gdf['Date'] = pd.to_datetime(formated_gdf['Date'])
formated_gdf['Date'] = formated_gdf['Date'].dt.strftime('%m/%d/%Y')


formated_gdf['Recovered_Cases'] = formated_gdf.Recovered +1

#Plotting the figure


fig = px.choropleth(formated_gdf, locations="Country", locationmode='country names', 
                     color="Recovered_Cases", hover_name="Country",projection="mercator",
                     animation_frame="Date",width=700, height=800,
                     color_continuous_scale=px.colors.sequential.Viridis,
                     title='COVID-19 Recovered Cases Across World')

#Showing the figure
fig.update(layout_coloraxis_showscale=True)
py.offline.iplot(fig)
#Creating the interactive map for death cases across the world
py.init_notebook_mode(connected=True)

#GroupingBy the dataset for the map
formated_gdf = df.groupby(['Date', 'Country'])['Deaths'].max()
formated_gdf = formated_gdf.reset_index()
formated_gdf['Date'] = pd.to_datetime(formated_gdf['Date'])
formated_gdf['Date'] = formated_gdf['Date'].dt.strftime('%m/%d/%Y')


formated_gdf['Fatalities'] = formated_gdf.Deaths + 1

#Plotting the figure


fig = px.choropleth(formated_gdf, locations="Country", locationmode='country names', 
                     color="Fatalities", hover_name="Country",projection="mercator",
                     animation_frame="Date",width=700, height=800,
                     color_continuous_scale=px.colors.sequential.Viridis,
                     title='COVID-19 Death Cases Across World')

#Showing the figure
fig.update(layout_coloraxis_showscale=True)
py.offline.iplot(fig)
bycountry = df.groupby(['Country'])['Confirmed','Deaths','Recovered'].max()
bycountry = bycountry.sort_values(['Confirmed','Deaths','Recovered'], ascending=False)
countrytop10 = bycountry.head(10)
countrytop10
top10plot = countrytop10.plot(kind='barh', stacked=True,figsize=(15,10),title='Top 10 Countries by Number of Cases',cmap='Spectral')
top10plot.set_xlabel('Number of cases')
top10plot.set_ylabel('Country');
usa_group = df.groupby(['Country','States'])['Confirmed'].max().get('US')
usa_group = usa_group.sort_values(ascending=False)
usa_top10 = usa_group.head(10)
usa_top10
usa_plot=plt.figure(figsize=(20,10))
usa_top10.plot(title='Top 10 US States by Confirmed Cases', kind='pie', fontsize=12, explode=(0.05,0,0,0,0,0,0,0,0,0), autopct='%1.1f%%', cmap='PuBuGn_r', pctdistance=0.6, labeldistance=1.1, shadow=True)
centre_circle = plt.Circle((0,0),0.70,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
plt.xlabel('')
plt.ylabel('')
plt.tight_layout()
plt.show();
aus_confirmed_group = df.groupby(['Country','States'])['Confirmed'].max().get('Australia')
aus_confirmed_group = aus_confirmed_group.sort_values(ascending=False)
aus_states_confirmed_group = aus_confirmed_group.drop(['From Diamond Princess','Jervis Bay Territory','External territories','Diamond Princess cruise ship'])
aus_states_confirmed_group
aus_plot=plt.figure(figsize=(20,10))
aus_states_confirmed_group.plot(title='Australian States by Confirmed Cases', kind='pie', fontsize=12, explode=(0,0,0.1,0,0,0,0,0), autopct='%1.1f%%', cmap="YlOrRd_r", pctdistance=1.1, labeldistance=1.2)
centre_circle = plt.Circle((0,0),0.70,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
plt.xlabel('')
plt.ylabel('')
plt.tight_layout()
plt.show();