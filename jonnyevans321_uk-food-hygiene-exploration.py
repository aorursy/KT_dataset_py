import numpy as np 

import pandas as pd 

import folium

from folium import IFrame



df = pd.read_csv("../input/poole2018.csv")
#colours for each marker

color_key={'5':'darkblue','4':'blue','3':'purple','2':'lightred','1':'darkred','0':'black','AwaitingInspection':'lightgray','Exempt':'lightgray'}



#initialise the map

map_hyg = folium.Map(location=[50.7299,-1.9615],zoom_start = 13) 



#add a marker to the map for each establishment

for index,row in df.iterrows():

    folium.Marker([row['Lat'], row['Long']],

                  popup= "<b>"+row['Name']+"</b> <br> Type: "+row['Type']+"<br> <br> Rating: "+row['Rating'],

                  icon=folium.Icon(color=color_key[row['Rating']])

                 ).add_to(map_hyg)

#show the map

map_hyg
from folium import plugins

from folium.plugins import HeatMap



#initialise the heatmap

heatmap = folium.Map(location=[50.7299,-1.9615],zoom_start = 13) 



# make the data for the heatmap by collecting the coordinates in a list

heat_data = [[row['Lat'],row['Long']] for index, row in df.iterrows()]



# Plot it on the map

HeatMap(heat_data).add_to(heatmap)



# Display the map

heatmap
#get counts for each rating for each est. type

counts=[]

for est_type in np.unique(df['Type']):

    est_type_data={}

    df_type=df[df['Type']==est_type].copy()

    for rating in np.unique(df_type['Rating']):

        est_type_data[rating]=(df_type.Rating == rating).sum()

    counts.append(est_type_data)

#create and show the resulting dataframe

df_counts=pd.DataFrame(counts,index=np.unique(df['Type'])).fillna(0)

df_counts
#plot a stacked bar chart of the dataframe

import seaborn as sns

from matplotlib.colors import ListedColormap



sns.set(rc={'figure.figsize':(30,15)},font_scale=3)

df_counts.iloc[:,0:5].plot(kind='bar', stacked=True,colormap=ListedColormap(sns.color_palette("GnBu", 5)),)
from folium import plugins 



#initialise the heatmap

heatmap = folium.Map(location=[50.7299,-1.9615],zoom_start = 13)



#colelct the data for the heatmap in a list of lists (one for each year)

heat_data=[]

years=np.arange(2013,2019)

for year in years:

    file="../input/poole"+str(year)+".csv"

    print(file)

    df = pd.read_csv(file)

    heat_data.append([[row['Lat'],row['Long']] for index, row in df.iterrows()])



# Plot it on the map

hm = plugins.HeatMapWithTime(heat_data,auto_play=True,max_opacity=0.8)

hm.add_to(heatmap)



# Display the map 

heatmap



#Note the buttons to control the heatmap are on the bottom right. Not sure why they arent visible, but if you hover over them the commands can be seen