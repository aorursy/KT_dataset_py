#Import modules 



import numpy as np

import pandas as pd

import random

import matplotlib.pyplot as plt

import seaborn as sns

import folium

from folium import plugins

from mpl_toolkits.basemap import Basemap

from matplotlib import animation,rc

from IPython.display import HTML, display

import io

import warnings

warnings.filterwarnings('ignore')

import codecs

import base64

import plotly.express as px

import plotly.figure_factory as ff

import plotly.graph_objects as go



plt.style.use("fivethirtyeight")

plt.rcParams['figure.figsize'] = (8, 6)
loading_columns=['ID', 'Severity', 'Start_Time', 'End_Time','Start_Lat', 'Start_Lng', 'City', 'County', 'State', 'Timezone']

df=pd.read_csv('../input/us-accidents/US_Accidents_June20.csv', usecols=loading_columns)
# Top states with the highest accidents

sns.countplot(df['State'], order=df['State'].value_counts().iloc[:10].index)

plt.xticks(rotation=0)

plt.title("Top 10 states with the most accidents", fontsize=25)

plt.tight_layout()
#convert datetime

df.Start_Time=pd.to_datetime(df.Start_Time)

df.End_Time=pd.to_datetime(df.End_Time)



#Plot the total accidents by years

sns.countplot(df['Start_Time'].dt.hour, hue=df['Severity'])

plt.xticks(rotation=0)

plt.title("Total accidents by hour", fontsize=25)

plt.tight_layout()
#Plot the total accidents by years

sns.countplot(df['Start_Time'].dt.month)

plt.xticks(rotation=0)

plt.title("Total accidents by month", fontsize=25)

plt.tight_layout()
Month_day=pd.crosstab(df['Start_Time'].dt.day, df['End_Time'].dt.month)



ax=sns.heatmap(Month_day,linewidths=.5, cmap='coolwarm')

ax.set_title("USA accidents (Month-Day)")

ax.set(xlabel='Month', ylabel='Day')

plt.tight_layout()

Week_day=pd.crosstab(df['Start_Time'].dt.hour, df['End_Time'].dt.dayofweek+1)

ax1=sns.heatmap(Week_day,linewidths=.5, cmap='coolwarm',)

ax1.set_title("USA accidents (Weekday - Hour)")

ax1.set(xlabel='Day of the week', ylabel='Hour')

plt.tight_layout()
#Visualize the coordinates

fig = plt.figure(figsize = (10,8))

df2=df[df['Severity']==3]

df2=df2.dropna(subset=['Start_Lat','Start_Lng'])



def animate(Hour):

    ax = plt.axes()

    ax.clear()

    ax.set_title('Accidents (severity=3) In USA '+'\n'+'Hour (Local time):' +str(Hour))

    m6 = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,

        projection='lcc',lat_1=33,lat_2=45,lon_0=-95)

    lat_gif1=list(df2[df2['Start_Time'].dt.hour==Hour].Start_Lat)

    long_gif1=list(df2[df2['Start_Time'].dt.hour==Hour].Start_Lng)

    x_gif1,y_gif1=m6(long_gif1,lat_gif1)

    m6.scatter(x_gif1, y_gif1, color='r') 

    m6.drawcoastlines()

    m6.drawcountries()

    m6.drawstates()

    m6.fillcontinents(color='coral',lake_color='aqua', zorder = 1,alpha=0.4)

    m6.drawmapboundary(fill_color='aqua')

ani = animation.FuncAnimation(fig,animate,list(sorted(df2['Start_Time'].dt.hour.unique())), interval = 1500)    

ani.save('animation_hour.gif', writer='imagemagick', fps=1)

plt.close(1)

filename = 'animation_hour.gif'

video = io.open(filename, 'r+b').read()

encoded = base64.b64encode(video)

HTML(data='''<img src="data:image/gif;base64,{0}" type="gif" />'''.format(encoded.decode('ascii')))
df['Accident_hour']=df['Start_Time'].dt.hour

df3=df.groupby(['Accident_hour', 'State'])['ID'].count()

df3=df3.reset_index()

df3.columns=['Accident_hour', 'State', 'Accident_counts']



fig=px.choropleth(data_frame=df3, locations='State', locationmode='USA-states', 

                  color='Accident_counts', animation_frame='Accident_hour', 

                  color_continuous_scale='Reds', 

                  color_continuous_midpoint=round(df3['Accident_counts'].max()/2, -3))



fig.update_layout(

    title_text = 'Accidents for different states at different hours',

    title_x=0.5,

    geo_scope='usa', # limite map scope to USA

)



fig.show()
#US state population

uspopulation=pd.read_csv("https://raw.githubusercontent.com/jakevdp/data-USstates/master/state-population.csv")

uspop_2013=uspopulation[(uspopulation['year']==2013) & (uspopulation['ages'] == 'total')]

uspop_2013=uspop_2013.drop(columns=['ages', 'year'])

uspop_2013.columns=['State', 'Population']



df4=df3.merge(uspop_2013)

df4['Accident_per_1k_resident']=df4['Accident_counts']/df4['Population']*1000



fig1=px.choropleth(data_frame=df4, locations='State', locationmode='USA-states', 

                  color='Accident_per_1k_resident', animation_frame='Accident_hour', 

                  color_continuous_scale='Reds')



fig1.update_layout(

    title_text = 'Accidents per 1k residents for different states at different hours',

    title_x=0.5,

    geo_scope='usa', # limite map scope to USA

)



fig1.show()
bay_area_counties=['Alameda' 'San Francisco', 'San Mateo', 'Santa Clara',]

CA=df[df["State"]=='CA']

bay_area=CA[CA['County'].isin(bay_area_counties)]

bay_area['Start_Time']=pd.to_datetime(bay_area['Start_Time'])



fig1 = px.density_mapbox(bay_area, lat='Start_Lat', lon='Start_Lng', radius=5, color_continuous_scale='Reds',

                        mapbox_style="stamen-terrain",)

fig1.update_layout(title = 'Bay Area Accidents Heatmap')



fig1.show()
bay_area['roundlat']=bay_area['Start_Lat'].round(3)

bay_area['roundlon']=bay_area['Start_Lng'].round(3)

hotspot_bayarea=bay_area.groupby(['roundlat', 'roundlon'])['ID'].count().sort_values(ascending=False).reset_index()[:50]

hotspot_bayarea.columns=['lat','lon','count']



m= folium.Map(location=[37.38, -122.08], zoom_start=10,)



for lat, lng, size, in zip(hotspot_bayarea.lat, hotspot_bayarea.lon, hotspot_bayarea['count']):

    folium.CircleMarker(

        location=[lat, lng],

        radius=size/20,

        color='red',

        fill=True,

        fill_color='yellow',

        fill_opacity=0.4

    ).add_to(m)



m
#What about the whole nations? Where are the most dangerous locations? 

df['lat'], df['lng']=df['Start_Lat'].round(4), df['Start_Lng'].round(4)

hotspot=df.groupby(['lat', 'lng'])['ID'].count().sort_values(ascending=False)[:20].reset_index()

hotspot.columns=['lat', 'lng', 'Count']





m2= folium.Map(location=[40, -102], zoom_start=4,)



for lat, lng, size, in zip(hotspot.lat, hotspot.lng, hotspot_bayarea['count']):

    folium.CircleMarker(

        location=[lat, lng],

        radius=size/20,

        popup=size,

        color='red',

        fill=True,

        fill_color='yellow',

        fill_opacity=0.4

    ).add_to(m2)



m2
df.groupby('Accident_hour')['Severity'].mean().plot(kind='line')

plt.xlabel('Hour of the day')

plt.ylabel('Average Severity')

plt.title('Average severity at different hours')

plt.tight_layout()
state_severity=df.groupby(['State', 'Accident_hour'])['Severity'].mean().reset_index()



fig3=px.choropleth(data_frame=state_severity, locations='State', locationmode='USA-states', 

                  color='Severity', animation_frame='Accident_hour', 

                  color_continuous_scale='Reds',)



fig3.update_layout(

    title_text = 'Average severity for different states at different hours',

    title_x=0.5,

    geo_scope='usa', # limite map scope to USA

)



fig3.show()
df['Accident_month']=df['Start_Time'].dt.month

df.groupby('Accident_month')['Severity'].mean().plot(kind='line')

plt.ylabel('Average Severity')

plt.title('Average Severity by Month')
state_month_severity=df.groupby(['State', 'Accident_month'])['Severity'].mean().reset_index()



fig4=px.choropleth(data_frame=state_month_severity, locations='State', locationmode='USA-states', 

                  color='Severity', animation_frame='Accident_month', 

                  color_continuous_scale='Reds',)



fig4.update_layout(

    title_text = 'Average severity for different states at different months',

    title_x=0.5,

    geo_scope='usa', # limite map scope to USA

)



fig4.show()