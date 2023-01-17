# import libraries



import numpy as np

import pandas as pd

import folium

import webbrowser

from folium.plugins import HeatMap
#This data is downloaded from Johns Hopkins CSSE shared Google sheet, unnecessary columns were removed

df = pd.read_csv("../input/wuhancsv/wuhan.csv")

df.head()
#data length

num=df.shape[0]

print(num)
%matplotlib inline

import matplotlib.pyplot as plt

df2=df[['Country/Region','Confirmed']]

df3=df2.groupby(['Country/Region']).sum().sort_values(by=['Confirmed'], ascending=False)

print(df3)
fig=plt.style.use('ggplot')

df3.plot(kind='bar',figsize=(10,6))

plt.ylabel('Confirmed')
# Get latitude



lat=np.array(df["Lat"][0:num])

# Get longitude

lon =np.array(df["Long"][0:num])



#Get confirmed cases 



confirmed=np.array(df["Confirmed"][0:num],dtype=float)



# [lat, long, weights]

data=[[lat[i],lon[i],confirmed[i]] for i in range(num)]



#map center[32,120], zoom levels = 5

map=folium.Map(location=[32,120], zoom_start=5)

# add heat map to the map we just created



HeatMap(data).add_to(map)

 

#file_path=r"wuhan.html"



# save it as html if you like



#map.save(file_path)

# open html with your browser



map
# In this example, with the hep of heat maps, we are able to perceive the density of confirmed cases

# another map

darkmap = folium.Map(location = [32,120], tiles='Cartodb dark_matter', zoom_start = 5)



HeatMap(data).add_to(darkmap)

 

darkmap