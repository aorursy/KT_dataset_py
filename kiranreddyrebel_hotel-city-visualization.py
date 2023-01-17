# My First Visualization.


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
#print(os.listdir(""))

# Any results you write to the current directory are saved as output.


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
%matplotlib inline

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999

import matplotlib.pyplot as plt
import matplotlib.cm

from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize
df = pd.read_csv('../input/makemytrip_com-travel_sample.csv',error_bad_lines=False)
df.describe()
# as we have many cities here , i am going to choose few cities only to try visualization on them

# selected top cities to create a new Dataframe
#Hyderabad,Kolkata,Guwahati,Mumbai,NewDelhiAndNCR,Bangalore,Vijaywada,Chennai

cities = ['Hyderabad','Kolkata','Guwahati','Mumbai','NewDelhiAndNCR','Bangalore','Vijaywada','Chennai']

newdf = df.loc[df['city'].isin(cities)]
# removing "star" char from hotel_star_rating column

newdf['hotel_star_rating']=newdf['hotel_star_rating'].replace('1 star',1).astype(str)
newdf['hotel_star_rating']=newdf['hotel_star_rating'].replace('2 star',2).astype(str)
newdf['hotel_star_rating']=newdf['hotel_star_rating'].replace('3 star',3).astype(str)
newdf['hotel_star_rating']=newdf['hotel_star_rating'].replace('4 star',4).astype(str)
newdf['hotel_star_rating']=newdf['hotel_star_rating'].replace('5 star',5).astype(str)
newdf['hotel_star_rating']=newdf['hotel_star_rating'].replace('Four star',4).astype(str)

# here as we can see that there are no. of duplicate entries for same 'Area' of particular city

newdf[newdf['area']=='Ballygunge']

# So , what i am doing is that, i will remove duplicates and keep only one entry for a area.
# and store in newdf itself


newdf = newdf.drop_duplicates(subset='area', keep="first")


# remove duplicate latitude and longitude

newdf = newdf.drop_duplicates(subset=['latitude','longitude'], keep="first")

# now crosstab to create DF to visualize

s = pd.crosstab(index=newdf['city'],columns=newdf['hotel_star_rating'])
bar = s.plot(kind='bar',figsize=(15,5),title='Count of Ratings based on Cities',fontsize=15)
bar
# Here we can see that NewDelhiAndNCR has top ratings on 3.
newdf1 = newdf['hotel_star_rating'].groupby(newdf['city']).count()
newdf1 = newdf1.reset_index()
newdf1.rename(columns ={'hotel_star_rating':'count'}, inplace = True)
newdf1 = newdf1.sort_values('count', ascending = False)
newdf1.reset_index(drop=True, inplace = True)
sns.set_context("poster", font_scale=0.6)
plt.rc('font', weight='bold')
f, ax = plt.subplots(figsize=(11, 6))
labels = [s[0] if s[1] > 80 else ' ' 
          for index, s in  newdf1[['city', 'count']].iterrows()]
sizes  = newdf1['count'].values
explode = [0.0 if sizes[i] < 100 else 0.0 for i in range(len(newdf1))]
ax.pie(sizes, explode = explode, labels = labels,
       autopct = lambda x:'{:1.0f}%'.format(x) if x > 1 else '',
       shadow=False, startangle=45)
ax.axis('equal')
ax.set_title('% of Visits (as per ratings in MakeMyTrip)  per City',
             bbox={'facecolor':'k', 'pad':5},color='w', fontsize=16);
# if there's any NaN in mmt_review_count, then fill with 0 because we don't want to give any review count ourself.

newdf.mmt_review_count = newdf.mmt_review_count.fillna(0)
newdf2 = newdf['city'].groupby(newdf.mmt_review_count)
kk = pd.crosstab(index=newdf['city'],columns=newdf['mmt_review_count'].sum())
kk.plot(kind='barh',title='Cities having Maximum Reviews', fontsize=14)
# Extracting latitude and longitude and area field to plot on Map

data = pd.DataFrame(newdf, columns = ['latitude','longitude','area']) #[:50]  <-this kind of slice is useful for developing a map
data.area = data.area.fillna('none')

# remove those rows where latitude & longitude is null
data = data.drop(data[data.latitude.isnull()].index)
data = data.drop(data[data.longitude.isnull()].index)

import folium
import sys

#reload(sys)                         # for python 2 only
#sys.setdefaultencoding('utf8')      # for python 2 only, in py3 it works by default

#create a map
this_map = folium.Map(prefer_canvas=True)

def plotDot(point):
    folium.CircleMarker(location=[point.latitude,point.longitude],
                        radius=5,
                        weight=2,
                        popup = point.area,
                        fill_color='red').add_to(this_map)

data.apply(plotDot, axis = 1)
this_map.fit_bounds(this_map.get_bounds())
# India Map with plot on cities (make my trip)
this_map
# Here we can see that some locations areas ar fake, they are pointing to outside of India

# Lets remove them
# action to remove that out of India locations..
data = data.drop(data[data.area=='Bapujinagar, Jadavpur'].index)
data = data.drop(data[data.area=='NEAR TANK NO - 9'].index)
data = data.drop(data[data.area=='Besides Peddamma Temple'].index)
data = data.drop(data[data.area=='Dollars Colony'].index)
data = data.drop(data[data.area=='HRBR layout, Bangalore'].index)
data = data.drop(data[data.area=='west Extension'].index)
data = data.drop(data[data.area=='201301'].index)
data = data.drop(data[data.area=='None'].index)
data = data.drop(data[data.area=='none'].index)
data = data.drop(data[data.area=='RMV Ext , Sanjaynagar'].index)
data = data.drop(data[data.area=='M.G. Road'].index)
# runned again to show correct plot on India map


import folium
import sys
#create a map
this_map = folium.Map(prefer_canvas=True)
def plotDot(point):
    folium.CircleMarker(location=[point.latitude,point.longitude],
                        radius=5,
                        weight=2,
                        popup = point.area,
                        fill_color='red').add_to(this_map)
data.apply(plotDot, axis = 1)
this_map.fit_bounds(this_map.get_bounds())
# correctly plotted maps from MakeMyTrip on India map
this_map
# New Delhi NCR MAp plot
# click on the Circle on map to show the Area name And Review Counts on that area hotel.
# per city wise map plot


delhimap = pd.DataFrame(newdf[newdf.city=='NewDelhiAndNCR'], columns = ['latitude','longitude','area','mmt_review_count']) #[:50]  <-this kind of slice is useful for developing a map
delhimap.area = data.area.fillna('none')
delhimap['area_review_count'] = delhimap['area']+'->ReviewCount: '+delhimap['mmt_review_count'].astype(str)


import folium
import sys
#create a map
this_map1 = folium.Map(prefer_canvas=True, zoom_start=20)
def plotDot1(point):
    folium.CircleMarker(location=[point.latitude,point.longitude],
                        radius=2,
                        weight=10,
                        popup = point.area_review_count,
                        fill_color='red').add_to(this_map1)
    
delhimap.apply(plotDot1, axis = 1)
this_map1.fit_bounds(this_map1.get_bounds())
this_map1
# Guwahati Map Plot 

# click on the Circle on map to show the Area name.

vijaywadamap = pd.DataFrame(newdf[newdf.city=='Guwahati'], columns = ['latitude','longitude','area','mmt_review_count']) #[:50]  <-this kind of slice is useful for developing a map
vijaywadamap.area = data.area.fillna('none')
vijaywadamap['area_review_count'] = vijaywadamap['area']+'->ReviewCount: '+vijaywadamap['mmt_review_count'].astype(str)


import folium
import sys
#create a map
this_map2 = folium.Map(prefer_canvas=True, zoom_start=20)
def plotDot1(point):
    folium.CircleMarker(location=[point.latitude,point.longitude],
                        radius=2,
                        weight=10,
                        popup = point.area_review_count,
                        fill_color='red').add_to(this_map2)
    
vijaywadamap.apply(plotDot1, axis = 1)
this_map2.fit_bounds(this_map2.get_bounds())
this_map2

# Which City has Highest number of Hotels
q = newdf['city'].value_counts().reset_index().iloc[0]['index']
print("Answer : ");q
# Convert all rating to int type

newdf['hotel_star_rating'] = newdf['hotel_star_rating'].astype(int)
# Interactive map of Hyderabad with location and ratings with Hotel name

maphotel = newdf.groupby(['property_name', 'latitude', 'longitude'])['hotel_star_rating'].mean().reset_index()
maphotel['name_rating'] = maphotel['property_name']+' ReviewCount: '+maphotel['hotel_star_rating'].astype(str)

lat = maphotel.latitude
lon = maphotel.longitude
name = maphotel.property_name
rating = maphotel.hotel_star_rating

mapbox_access_token = 'pk.eyJ1Ijoia2FtcGFyaWEiLCJhIjoib0JLTExtSSJ9.6ahf835RV3kBUnC3cQ-SnA'
data = go.Data([
    go.Scattermapbox(
        lat=lat,
        lon=lon,
        mode='markers',
        marker=go.Marker(
            size=10,
            color='rgb(255, 0, 0)',
            opacity=0.7
        ),
        text=maphotel.property_name,
        hoverinfo=maphotel.name_rating
    ),
    go.Scattermapbox(
        lat=lat,
        lon=lon,
        mode='markers',
        marker=go.Marker(
            size=8,
            color='rgb(242, 177, 172)',
            opacity=0.7
        ),
        text=maphotel.name_rating,
        hoverinfo=''
    )]
)
        
layout = go.Layout(
    title='Interactive Map Showing The Location Of Hotel and Name & Ratings.',
    autosize=True,
    hovermode='closest',
    showlegend=False,
    mapbox=dict(
        accesstoken=mapbox_access_token,
        bearing=0,
        center=dict(
            lat=17.387140,
            lon=78.491684
        ),
        pitch=0,
        zoom=12,
        style='dark'
    ),
)

fig = dict(data=data, layout=layout)

py.iplot(fig, filename='hotelreview')

