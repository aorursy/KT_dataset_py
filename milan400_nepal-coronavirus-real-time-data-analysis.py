#Import pandas to convert list to data frame



import pandas as pd

import numpy as np

import requests

from pandas_profiling import ProfileReport



import urllib.request



#import the beatiful soup functions to parse the data

from bs4 import BeautifulSoup





from plotly.offline import iplot, init_notebook_mode

import plotly.express as px

import plotly.graph_objects as go



import cufflinks as cf

cf.go_offline()

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        pass

        #print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

from IPython.display import display, Image





import folium

from folium import Choropleth, Circle, Marker

from folium.plugins import HeatMap, MarkerCluster



import geopandas as gpd

from geopandas import GeoDataFrame

from shapely.geometry import Point,Polygon



import math





URL = "https://coronanepal.live/"

#query the website

page = requests.get(url = URL)



#parse the html and store in Beautiful soup format

soup = BeautifulSoup(page.text)



#find all links



all_links = soup.find_all("a")

for link in all_links:

    pass

    #print(link.get("href"))

    



#find all tables

all_tables = soup.find('table')

#print(all_tables)





#Generate lists



A = []

B = []

C = []

D = []

E = []

F = []



for row in all_tables.findAll("tr"):

    cells = row.findAll('td')

    

    #Only extract table body

    if(len(cells) == 6):

        A.append(cells[0].find(text = True))

        B.append(cells[1].find(text = True))

        C.append(cells[2].find(text = True))

        D.append(cells[3].find(text = True))

        E.append(cells[4].find(text = True))

        F.append(cells[5].find(text = True))

        



df = pd.DataFrame()



df['जिल्ला'] = A

df['District'] = B

df['Confirmed Cases'] = C

df['कुल संक्रमित'] = D

df['जम्मा मृत्यु'] = E

df['निको भएको'] = F



df = df.astype({'जिल्ला':str,'District':str,'जम्मा मृत्यु':int, 'निको भएको':int})

df_nepal = df.astype({'जिल्ला':str,'District':str,'Confirmed Cases':int,'जम्मा मृत्यु':int, 'निको भएको':int})

df_nepal['हाल बिरामीहरूको संख्या'] = df_nepal['Confirmed Cases'] - df_nepal['निको भएको']



df_nepal_confirmed = df_nepal[['जिल्ला', 'कुल संक्रमित']]

df_nepal_dead = df_nepal[['जिल्ला', 'जम्मा मृत्यु']]

df_nepal_recovered = df_nepal[['जिल्ला','निको भएको']]



df_nepal_dict = dict()

df_nepal_dict['Count_For'] = "Nepal"

df_nepal_dict['कुल संक्रमित'] = df_nepal['कुल संक्रमित'].sum(axis=0)

df_nepal_dict['जम्मा मृत्यु'] = df_nepal['जम्मा मृत्यु'].sum(axis=0)

df_nepal_dict['निको भएको'] = df_nepal['निको भएको'].sum(axis=0)

df_nepal_dict['हाल बिरामीहरूको संख्या'] = df_nepal['हाल बिरामीहरूको संख्या'].sum(axis=0)



df_nepal_dataframe = pd.DataFrame.from_dict(df_nepal_dict, orient='index')





from IPython.display import display, HTML

display(HTML(df.to_html()))



#save data nepal

df_nepal_dataframe.to_csv('district_wise_data.csv', index=False)
URL = "https://pomber.github.io/covid19/timeseries.json"



# sending get request and saving the response as response object 

r = requests.get(url = URL) 



# extracting data in json format 

data = r.json() 



nepal_data = data['Nepal']



date_list = []

confirmed_list = []

recovered_list = []

death_list = []



for i in range(len(nepal_data)):

    date_list.append(nepal_data[i]['date'])

    confirmed_list.append(nepal_data[i]['confirmed'])

    recovered_list.append(nepal_data[i]['recovered'])

    death_list.append(nepal_data[i]['deaths'])

    

nepal_data_frame = pd.DataFrame()  

nepal_data_frame['Date'] = date_list

nepal_data_frame['Confirmed'] = confirmed_list

nepal_data_frame['Recovered'] = recovered_list

nepal_data_frame['Death'] = death_list



nepal_data_frame['Date'] = pd.to_datetime(nepal_data_frame['Date'])

nepal_data_frame['Date'] = nepal_data_frame['Date'].apply(lambda x:x.strftime('%Y-%m-%d'))





#nepal_data_frame['Date'] = nepal_data_frame['Date'].apply(pd.to_datetime)

nepal_data_frame = nepal_data_frame.astype({'Confirmed':int, "Recovered":int})



#save timeseries data nepal

nepal_data_frame.to_csv('nepal_timeseries.csv', index=False)



import plotly.graph_objects as go



fig = go.Figure()

fig.add_trace(go.Scatter(x=nepal_data_frame['Date'], y=nepal_data_frame['Confirmed'], fill='tozeroy',name='Confirmed Case'))

fig.add_trace(go.Scatter(x=nepal_data_frame['Date'], y=nepal_data_frame['Death'], fill='tozeroy',name='Death Case'))

fig.add_trace(go.Scatter(x=nepal_data_frame['Date'], y=nepal_data_frame['Recovered'], fill='tozeroy',name='Recovered Case'))





# Set x-axis title

fig.update_xaxes(title_text="Year")

fig.update_yaxes(title_text="Number of Cases")

fig.show()
nepal_data_frame_con = pd.DataFrame(columns=['Date','Case','Case_type'])

nepal_data_frame_recov = pd.DataFrame(columns=['Date','Case','Case_type'])

nepal_data_frame_death = pd.DataFrame(columns=['Date','Case','Case_type'])



confirmed_cases = nepal_data_frame['Confirmed']

recovered_cases = nepal_data_frame['Recovered']

death_cases = nepal_data_frame['Death']

date_ = nepal_data_frame['Date']



def parallel_add(df,date,a,b,case_,bdata):

    df[a] = date_

    df[b] = bdata

    df['Case Type']= case_

    return df



nepal_data_frame_con = parallel_add(nepal_data_frame_con,date_,'Date','Number of Cases','Confirmed' ,confirmed_cases)

nepal_data_frame_recov = parallel_add(nepal_data_frame_recov,date_,'Date','Number of Cases','Recovered' ,recovered_cases)

nepal_data_frame_death = parallel_add(nepal_data_frame_death,date_,'Date','Number of Cases','Death' ,death_cases)



nepal_data_total = nepal_data_frame_con.copy()

nepal_data_total = nepal_data_total.append(nepal_data_frame_recov, ignore_index = True)

nepal_data_total = nepal_data_total.append(nepal_data_frame_death, ignore_index = True)
fig = px.bar(nepal_data_total, x="Case Type", y="Number of Cases", color="Case Type",

  animation_frame="Date", animation_group="Case Type", range_y=[0,max(confirmed_cases)])

fig.show()
df_nepal_dataframe = df_nepal_dataframe.T



df_nepal_dataframe.iplot(kind='bar',x='Count_For', y =['कुल संक्रमित', 'जम्मा मृत्यु', 'निको भएको','हाल बिरामीहरूको संख्या'])
labels = df_nepal['District'].tolist()

values = df_nepal['जम्मा मृत्यु'].tolist()



if(sum(values) >0):

    layout = go.Layout(

        autosize=True,

        title = 'CoronaVirus Death Cases in Nepal',

        xaxis= go.layout.XAxis(linecolor = 'black',

                              linewidth = 1,

                              mirror = True),



        yaxis= go.layout.YAxis(linecolor = 'black',

                              linewidth = 1,

                              mirror = True),

    )

    fig = go.Figure(data=[go.Pie(labels = labels, values=values, hole=0.3, textposition='inside')])

    fig.update(layout=layout)

    fig.show()

else:

    print('No Death Case in Nepal')

labels = df_nepal['District'].tolist()

values = df_nepal['निको भएको'].tolist()



layout = go.Layout(

    autosize=True,

    title = 'CoronaVirus Recovered Cases in Nepal by district',

    xaxis= go.layout.XAxis(linecolor = 'black',

                          linewidth = 1,

                          mirror = True),



    yaxis= go.layout.YAxis(linecolor = 'black',

                          linewidth = 1,

                          mirror = True),

)

fig = go.Figure(data=[go.Pie(labels = labels, values=values, hole=0.3, textposition='inside')])

fig.update(layout=layout)

fig.show()

labels = df_nepal['District'].tolist()

values = df_nepal['हाल बिरामीहरूको संख्या'].tolist()



layout = go.Layout(

    autosize=True,

    title = 'CoronaVirus Active Cases in Nepal by district',

    xaxis= go.layout.XAxis(linecolor = 'black',

                          linewidth = 1,

                          mirror = True),



    yaxis= go.layout.YAxis(linecolor = 'black',

                          linewidth = 1,

                          mirror = True),

)

fig = go.Figure(data=[go.Pie(labels = labels, values=values, hole=0.3, textposition='inside')])

fig.update(layout=layout)

fig.show()

#Preproceesing Data for Choropleth maps

districts_full = gpd.read_file('/kaggle/input/nepal-data-location/NPL_adm3.shx')



districts_full_geometry = districts_full[['NAME_3', 'geometry']]



districts_full_geometry_dict = dict()

for key,value in zip(districts_full_geometry['NAME_3'],districts_full_geometry['geometry']):

    districts_full_geometry_dict[key.lower()] = value





def return_coordinates(x):

    return districts_full_geometry_dict[x.lower()]





df_nepal['District'] = df_nepal['District'].replace({'rukum2':'rukum','illam':'ilam','terathum':'terhathum',"rukum1":"rukum","chitwan": "chitawan", "dhanusha":"Dhanusa",'nawalparasi2':'nawalparasi', 'nawalparasi1':'nawalparasi','kavre':'Kavrepalanchok','sindhupalchowk':'Sindhupalchok'})



df_nepal['geometry'] = df_nepal['District'].apply(lambda x: return_coordinates(x))





df_nepal = df_nepal.reindex(df_nepal.index.repeat(df_nepal['हाल बिरामीहरूको संख्या']))



plot_dict = df_nepal.District.value_counts()



geometry = df_nepal['geometry']

df_nepal = df_nepal.drop(['जिल्ला','कुल संक्रमित', 'जम्मा मृत्यु', 'निको भएको','हाल बिरामीहरूको संख्या'], axis=1)

crs = {'init': 'epsg:4326'}

nepal_district_cases_map_gdf = GeoDataFrame(df_nepal, crs=crs, geometry=geometry)



nepal_district_cases_map_gdf = nepal_district_cases_map_gdf[['District','geometry']]

nepal_district_cases_map_gdf = nepal_district_cases_map_gdf.set_index('District')



df_nepal_map_point = nepal_district_cases_map_gdf.copy()

df_nepal_map_point['geometry'] = df_nepal_map_point['geometry'].centroid



df_nepal_map_point['Longitude'] = df_nepal_map_point.geometry.x

df_nepal_map_point['Latitude'] = df_nepal_map_point.geometry.y





# Create the map

m_3 = folium.Map(location=[28.3949,84.1240], tiles='cartodbpositron', zoom_start=7)



# Add points to the map

mc = MarkerCluster()

for idx, row in df_nepal_map_point.iterrows():

    if not math.isnan(row['Longitude']) and not math.isnan(row['Latitude']):

        mc.add_child(Marker([row['Latitude'], row['Longitude']]))

m_3.add_child(mc)



# Display the map

m_3