# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 









import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import scipy as sci

import itertools

from itertools import repeat,chain

from scipy.stats import boxcox

from pathlib import Path

import operator

from collections import namedtuple





####<-Visualization Libraries->####



import holoviews as hv

import matplotlib.pyplot as plt

import datashader

import dask

import geoviews



# Standard plotly imports

import chart_studio.plotly as py

from plotly.subplots import make_subplots

import plotly.graph_objs as go

import plotly.express as px

import geopandas

import shapely

import plotly.graph_objects as go



from plotly.subplots import make_subplots

from plotly.offline import iplot, init_notebook_mode



# Using plotly + cufflinks in offline mode

import cufflinks

cufflinks.go_offline(connected=True)

init_notebook_mode(connected=True)



####---------------------------####





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
## Importing dataset

data = pd.read_csv("/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv",sep=",")

data.head(5)
print("The dataset contains {0} rows and {1} columns  ".format(str(data.shape[0]),str(data.shape[1])))
print("Data Types: ")

print(data.dtypes)
# Checking missing values

print("Missing Values: ")

print(data.isnull().sum().sort_values(ascending=False))

print()

print("Do reviews_per_month and last_review share the same indices? "+ str(any(data.loc[data.last_review.isna(),"reviews_per_month"].index)))

print()
# Filling missing data

data["reviews_per_month"].fillna(0,inplace=True)

data["last_review"].fillna(0,inplace=True)

data[["host_name","name"]].fillna("Unknown",inplace=True)
# Grouping by neighbourhood_group and room_type

data_neighgroup_roomtype=data.groupby(["room_type","neighbourhood_group"])["neighbourhood_group"].count()





## PIE CHART ##

labels = data.neighbourhood_group.value_counts().index

values = data.neighbourhood_group.value_counts().values

fig_pie = go.Figure(data=[go.Pie(labels=labels, values=values)],layout=go.Layout(title=go.layout.Title(text="Number of dwellings per neighbourhood")))

fig_pie.show()





## BARPLOT ##

x=data_neighgroup_roomtype.index.levels[1]

room_type=data_neighgroup_roomtype.index.levels[0]

fig_bar = go.Figure(go.Bar(x=x, y=data_neighgroup_roomtype["Entire home/apt"], name=room_type[0]),layout=go.Layout(title=go.layout.Title(text="Room type per Neighbourhood")))

fig_bar.add_trace(go.Bar(x=x, y=data_neighgroup_roomtype["Private room"], name=room_type[1]))

fig_bar.add_trace(go.Bar(x=x, y=data_neighgroup_roomtype["Shared room"], name=room_type[2]))

fig_bar.update_layout(title="Number of apartments",barmode='group', xaxis={'categoryorder':'category ascending'})

neigh_room_review_mean =data.groupby(["room_type","neighbourhood_group"])["number_of_reviews"]

neigh_room_review_mean = neigh_room_review_mean.agg({"mean":"mean"})

neigh_room_review_mean
nrrm=neigh_room_review_mean.values.reshape(3,5)

room_type=['Entire home/apt', 'Private room', 'Shared room']



fig = go.Figure(data=[

    go.Bar(name='Bronx', x=room_type, y=nrrm[:,0]),

    go.Bar(name='Brooklyn', x=room_type, y=nrrm[:,1]),

    go.Bar(name='Manhattan', x=room_type, y=nrrm[:,2]),

    go.Bar(name='Queens', x=room_type, y=nrrm[:,3]),

    go.Bar(name='Staten Island', x=room_type, y=nrrm[:,4])





])

fig.update_layout(title="Mean of number of reviews",barmode='group')

fig.show()

price_loc_room = data.groupby(["neighbourhood_group","room_type"])["price"]

price_loc  = data.groupby(["neighbourhood_group"])["price"]

price_room = data.groupby(["room_type"])["price"]
price_loc_room.describe()
# Removing prices = 0

data.drop(data[data["price"]==0].index,inplace=True)

price_loc  = data.groupby(["neighbourhood_group"])["price"]

price_room = data.groupby(["room_type"])["price"]

price_loc_room = data.groupby(["neighbourhood_group","room_type"])["price"]
price_neighgroup_room=data.groupby(["room_type","neighbourhood_group"])["price"]

fig = go.Figure()



for room in iter(room_type):

    list_price=[]

    list_neigh=[]

    for neigh in iter(x):

        list_price.append(price_neighgroup_room.get_group((room,neigh)).values)

        list_neigh.append(list(repeat(neigh,len(price_neighgroup_room.get_group((room,neigh)).values))))



    list_price=np.concatenate(list_price).tolist()

    list_neigh = list(itertools.chain.from_iterable(list_neigh))

    

    fig.add_trace(go.Box(

    y=list_price,

    x=list_neigh,

    name=room))

    

    



fig.update_layout(

    title='Boxplot - Price by neighbourhood and room type',

    boxmode='group', # group together boxes of the different traces for each value of x

    yaxis_title="$"

)

fig.show()

    
fig = go.Figure()

fig.add_trace(go.Histogram(x=price_loc.get_group("Bronx"),name="Bronx"))

fig.add_trace(go.Histogram(x=price_loc.get_group("Brooklyn"),name="Brooklyn"))

fig.add_trace(go.Histogram(x=price_loc.get_group("Manhattan"),name="Manhattan"))

fig.add_trace(go.Histogram(x=price_loc.get_group("Queens"),name="Queens"))

fig.add_trace(go.Histogram(x=price_loc.get_group("Staten Island"),name="Staten Island"))



# Overlay both histograms

fig.update_layout(title="Price distribution by the location",barmode='overlay',xaxis_title="Price")

# Reduce opacity to see both histograms

fig.update_traces(opacity=0.75)

fig.show()
fig = go.Figure()

fig.add_trace(go.Histogram(x=price_room.get_group("Entire home/apt"),name="Entire home/apt"))

fig.add_trace(go.Histogram(x=price_room.get_group("Private room"),name="Private room"))

fig.add_trace(go.Histogram(x=price_room.get_group("Shared room"),name="Shared room"))



# Overlay both histograms

fig.update_layout(title="Price distribution by the room type",barmode='overlay',xaxis_title="Price")

# Reduce opacity to see both histograms

fig.update_traces(opacity=0.75)

fig.show()
high_prices=pd.DataFrame(columns={})

exceed_price=[]



for room in iter(room_type):

    for neigh in iter(x):

        median = np.median(price_loc_room.get_group((neigh,room)))

        upper_quartile = np.percentile(price_loc_room.get_group((neigh,room)), 75)

        lower_quartile = np.percentile(price_loc_room.get_group((neigh,room)), 25)



        iqr = upper_quartile - lower_quartile # Interquartile range

        upper_whisker = price_loc_room.get_group((neigh,room))[price_loc_room.get_group((neigh,room))<=upper_quartile+1.5*iqr].max()

        

        # Now we get the number of rooms (type "room") in the neighbourhood ("neigh") higher than the upper fence 

        apartments_high_than_upperwhisker = price_loc_room.get_group((neigh,room))[price_loc_room.get_group((neigh,room))>upper_whisker].count()

        

        exceed_price.append(apartments_high_than_upperwhisker)



exceed_price=np.array(exceed_price).reshape(3,5)

room_type=['Entire home/apt', 'Private room', 'Shared room']



fig = go.Figure(data=[

    go.Bar(name='Bronx', x=room_type, y=exceed_price[:,0]),

    go.Bar(name='Brooklyn', x=room_type, y=exceed_price[:,1]),

    go.Bar(name='Manhattan', x=room_type, y=exceed_price[:,2]),

    go.Bar(name='Queens', x=room_type, y=exceed_price[:,3]),

    go.Bar(name='Staten Island', x=room_type, y=exceed_price[:,4])





])

fig.update_layout(title="Number of apartments higher than the upper fence of the boxplot",barmode='group')

fig.show()

# The boxcox() SciPy function implements the Box-Cox method. 

# It takes an argument, called lambda, that controls the type of transform to perform.

# lambda = 0 is a log transformation

def log_transform(group):

    return boxcox(group,0)





data_log=data

data_log.price=data_log.price.apply(log_transform)

price_loc_log  = data_log.groupby(["neighbourhood_group"])["price"]

price_room_log = data_log.groupby(["room_type"])["price"]

price_loc_room_log = data.groupby(["neighbourhood_group","room_type"])["price"]

coordinates_and_price = zip(data.latitude,data.longitude,data.neighbourhood_group,data.price)

coordinates_and_price = sorted(coordinates_and_price,key=operator.itemgetter(3))

latitude   = list(zip(*coordinates_and_price))[0] 

longitude  = list(zip(*coordinates_and_price))[1] 

neighbours = list(zip(*coordinates_and_price))[2]

price      = list(zip(*coordinates_and_price))[3]
## With this cell you can select the neighbourhood you want to plot



# zone = str(input("What neighbourhood do you want to see?:\n\n0- Bronx\n1- Brooklyn\n2- Manhattan\n3- Queens\n4- Staten Island\n5- All\n"))



zone="All" # By default



if(zone!="All"):

    index=[]

    for i,j in enumerate(neighbours):

        if(j==zone):

            index.append(i)

else:

    index=np.arange(len(data))



mapbox_access_token = "pk.eyJ1IjoicG9sb2RlYWx2YXJhZG8iLCJhIjoiY2sycnp4N2YzMGF0OTNtcjZpMmhieDRhNSJ9.HmhtvNNEFy07YqjH4AHKSQ"

px.set_mapbox_access_token(mapbox_access_token)





fig = go.Figure(go.Scattermapbox(

        lat=list(operator.itemgetter(*index)(latitude)) ,

        lon=list(map(longitude.__getitem__, index)) ,

        mode='markers',

        marker=go.scattermapbox.Marker(

            size=10,

            color=list(map(price.__getitem__, index)),

            colorscale="Earth",

             colorbar=dict(

             title="Price",

             tickmode="array",             

             ticks="outside",

             tickvals=[10,100,500,2000,8000,10000] ,  

             ticktext=["10 $","100 $","500 $","2000 $","8000 $","10000 $"],

    

             

        )

   

        

    )))



fig.update_layout(

    title="Price in "+zone+" neighbourhood",

    autosize=True,

    hovermode='closest',

    mapbox_style="open-street-map",

    mapbox=go.layout.Mapbox(

        accesstoken=mapbox_access_token,

        bearing=0,

        center=go.layout.mapbox.Center(

            lat=40.64749,

            lon=-73.97237

        ),

        pitch=0,

        zoom=10,

    )

)





fig.show()