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
from IPython.display import Image

Image(filename='/kaggle/input/korea-coronavirus-additional-files/SKmap.gif') 
import pandas    as pd

import geopandas as gpd

import geopy     as gpy

import folium

from   folium import plugins

import datetime

import numpy as np



print(folium.__version__)
df_time           = pd.read_csv('/kaggle/input/coronavirusdataset/time.csv')

df_peroid         = df_time[df_time["date"] >= '2020-02-18']

df_peroid["rate"] = df_peroid["confirmed"]/df_peroid["test"] * 100 

df_peroid.tail(5)
import seaborn as sns

import matplotlib.pyplot as plt



plt.figure(figsize=(10,5))

barplot = sns.barplot(x=df_peroid['date'], y=df_peroid['rate'], palette="rocket")

plt.xticks(rotation=45)



plt.show()
def wordcloud_column(dataframe):

    from wordcloud import WordCloud, STOPWORDS 

 

    comment_words = ' '

    stopwords = set(STOPWORDS) 

  

    # iterate through the csv file 

    for k in range(len(dataframe)):

        # typecaste each val to string 

        val = str(dataframe.iloc[k,0]) 

        # split the value 

        tokens = val.split()

    

        # Converts each token into lowercase 

        for i in range(len(tokens)): 

            tokens[i] = tokens[i].lower() 

          

        for words in tokens: 

            comment_words = comment_words + words + ' '

  

    # lower max_font_size

    wordcloud = WordCloud(width=300, height=120,background_color ='white', max_font_size=60).generate(comment_words)

    plt.figure()

    plt.imshow(wordcloud, interpolation="bilinear")

    plt.axis("off")

    plt.show()
# Find the word cloud for Reason

df_patient = pd.read_csv('/kaggle/input/coronavirusdataset/patient.csv')

df_reason  = df_patient[['infection_reason']]

df_reason  = df_reason[(df_reason['infection_reason'].notna())]

wordcloud_column(df_reason)
# Find the word cloud for Group

df_group  = df_patient[['group']]

df_group  = df_group[(df_group['group'].notna())]

wordcloud_column(df_group) 
# read the region coordinates from region.csv

df_region = pd.read_csv('/kaggle/input/korea-coronavirus-additional-files/region.csv')



# Prepare for a map to show the recent confirmed cases



list = df_region['region'].tolist()               # generate a list of regions based on the region dataframe

result = df_time[list].iloc[-1].sort_index()      # get the most recent row sorted with regions.  

# Store the most recent counts on a new total

for col in result.index:

    df_region.loc[df_region['region'] == col, 'total'] = result[col]

                                

df_region.head(5)
# Create a legend

legend_html = '''

        <div style="position: fixed; bottom: 100px; left: 50px; width: 160px; height: 110px; 

                    background-color: white; border:2px solid grey; z-index:9999; font-size:14px;"

                    >&nbsp; <b>Legend</b> <br>

                    &nbsp; Confirmed < 100 &nbsp&nbsp&nbsp; 

                        <i class="fa fa-circle" style="font-size:14px;color:#ff9900"></i><br>

                    &nbsp; Confirmed < 1000 &nbsp; 

                        <i class="fa fa-circle" style="font-size:14px;color:#cc33ff"></i><br>

                    &nbsp; Confirmed < 3000 &nbsp; 

                        <i class="fa fa-circle" style="font-size:14px;color:#ff0000"></i><br>

                    &nbsp; Confirmed >= 3000

                        <i class="fa fa-circle" style="font-size:14px;color:#660000"></i>

        </div>

        ''' 



def color(total):

    # Color range

    col_100  = "#ff9900"

    col_1000 = "#cc33ff"

    col_3000 = "#ff0000"

    over     = "#660000"

    if (total < 100):   

            rad = total/10

            color = col_100

    elif (total < 1000): 

            rad = min(total/10, 20)

            color = col_1000

    elif (total < 3000): 

            rad = min(total/10, 30)

            color = col_3000

    else: 

            rad = 35

            color = over

    return rad, color

map0 = folium.Map(location=[35.7982008,125.6296572], control_scale=True, tiles='OpenStreetMap', zoom_start=7)

folium.TileLayer('openstreetmap').add_to(map0)

folium.TileLayer('CartoDB positron',name='Positron').add_to(map0)

folium.TileLayer('CartoDB dark_matter',name='Dark Matter').add_to(map0)

folium.TileLayer('Stamen Terrain',name='Terrain').add_to(map0)

folium.TileLayer('Stamen Toner',name='Toner').add_to(map0)

# Enable the layer control 

folium.LayerControl().add_to(map0)

# Enable Expand fullscreen feature

plugins.Fullscreen( position='topleft', title='Expand', title_cancel='Exit', force_separate_button=True ).add_to(map0) 

map0.get_root().html.add_child(folium.Element(legend_html))



for index, row in df_region.iterrows():

    

    total = row["total"]

    reg   = row["region"]

    lat   = row["latitude"]

    long  = row["longitude"]

    

    # generate the popup message that is shown on click.

    popup_text = "<b>Region:</b> {}<br><b>Confirmed: </b>{}"

    popup_text = popup_text.format(reg, total)          

    

    # select colors and radius

    rad, col = color(total)

    folium.CircleMarker(location=(lat,long), radius = rad, color=col, popup=popup_text, 

                        opacity= 4.0, fill=True).add_to(map0)



display(map0)
from folium.plugins import TimestampedGeoJson



map2 = folium.Map(location=[35.7982008,125.6296572], zoom_start=7, control_scale=True,tiles='CartoDB dark_matter')

folium.TileLayer('openstreetmap').add_to(map2)

folium.TileLayer('CartoDB positron',name='Positron').add_to(map2)

folium.TileLayer('CartoDB dark_matter',name='Dark Matter').add_to(map2)

folium.TileLayer('Stamen Terrain',name='Terrain').add_to(map2)

folium.TileLayer('Stamen Toner',name='Toner').add_to(map2)

# Enable the layer control 

folium.LayerControl().add_to(map2)

# Enable Expand fullscreen feature

plugins.Fullscreen( position='topleft', title='Expand', title_cancel='Exit', force_separate_button=True ).add_to(map2) 

map2.get_root().html.add_child(folium.Element(legend_html))



features = []

for index, row in df_time[(df_time['date']>='2020-02-18')].iterrows():

        # Extract the province counts 

        province = df_time[list].iloc[index].sort_index() 

        date = row['date']

        for i in range(len(province)):

            total  = province[i]

            if (total > 0):

                # select various colors and radius

                rad, col = color(total)

                lat    = df_region.at[i, 'latitude']

                long   = df_region.at[i, 'longitude']

                reg    = df_region.at[i, 'region']

                popup_text = "<b>Region:</b> {}<br><b>Confirmed: </b>{}"

                popup_text = popup_text.format(reg, total)

                feature = {

                    'type': 'Feature',

                    'geometry': {

                        'type':'Point', 'coordinates':[long, lat]},

                    'properties': {

                        'time': date.__str__(),

                        'style': {'color' : col},

                        'popup': popup_text,

                        'icon': 'circle',

                        'iconstyle':{

                            'fillColor': col,

                            'fillOpacity': 0.8,

                            'fill': 'true',

                            #'stroke': 'true',

                            'radius': rad}

                            }

                        }

                features.append(feature)



TimestampedGeoJson(

        {'type': 'FeatureCollection',

        'features': features}

        , period='P1D'

        , add_last_point=True

        , auto_play=False

        , loop=False

        , max_speed=1

        , loop_button=True

        , date_options='YYYY-MM-DD'

        , time_slider_drag_update=True).add_to(map2)



map0.save('SKConfirmedwithTimestamp.html')

display(map2)
# Define a zoom_prov function

def zoom_prov(province_in, zoomstart=12):

    province = df_route[df_route["province"] == province_in]

    # Initialize the area

    

    init_points  = (np.average(province.iloc[:,5]),np.average(province.iloc[:,6]))

    label_points = (np.average(province.iloc[:,5])-0.23,np.average(province.iloc[:,6])-0.23)



    temp_map     = folium.Map(location=init_points, zoom_start=zoomstart, control_scale=True,tiles='CartoDB Positron')

    plugins.Fullscreen( position='topleft', title='Expand', title_cancel='Exit', force_separate_button=True ).add_to(temp_map)



    # Create a City Name

    name1 = '''

        <div style="position: fixed; bottom: 50px; left: 50px; width: 130px; height: 65px; 

                    background-color: white; border:2px solid grey; z-index:9999; font-size:14px;"

                    >&nbsp; <br>  <b>Province: '''  

    name2 =  province_in + ' </b> </div> '

    name  = name1 + name2

    temp_map.get_root().html.add_child(folium.Element(name))

    

    for index, row in province.iterrows():

        date  = row["date"]

        visit = row["visit"]

        city  = row["city"]

        loc   = (row["latitude"], row["longitude"])

     

        # generate the popup message that is shown on click.

        popup_text = "<b>Date:</b> {}<br><b>Vist</b>: {}<br><b>City:</b> {}<br><b>Loc:</b> {}"

        popup_text = popup_text.format(date,visit,city,loc)

        icon       = folium.Icon(color='red', icon='info-sign')

        popup = folium.Popup(popup_text, max_width=300, min_width=80)

        folium.Marker(loc, popup=popup, icon=icon).add_to(temp_map)

    display(temp_map)
# Look into the buildings involved virus in Daegu

df_route     = pd.read_csv('/kaggle/input/coronavirusdataset/route.csv')

province = "Daegu"

zoom_prov(province)
province="Gyeongsangbuk-do"

zoom_prov(province, zoomstart=8)
# Look into what buildings got involved with virus in Seoul

province="Seoul"

zoom_prov(province)
# Look into what buildings got involved with virus in Gyeonggi-do 

province = "Gyeonggi-do"

zoom_prov(province,zoomstart=9)