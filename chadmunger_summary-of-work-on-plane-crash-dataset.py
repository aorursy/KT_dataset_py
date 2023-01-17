from IPython.display import IFrame

import pandas as pd 

import folium



IFrame('https://plotly.com/~Daedalus1/9/#/', width=700, height=350)
IFrame('https://plotly.com/~Daedalus1/13/#/', width=700, height=350)

IFrame('https://plotly.com/~Daedalus1/11/#/', width=700, height=350)
IFrame('https://plotly.com/~Daedalus1/16/', width=700, height=350)
sorted_working_data = pd.read_csv('../input/interactivemapdataset/InteractiveMapData.csv')



# Make an empty map

interactive_map = folium.Map(location=[0,10],

zoom_start=1,

tiles='Stamen Terrain')



# I can add marker one by one on the map

for i in range(0,len(sorted_working_data)):

    folium.Circle(

      location=[sorted_working_data.iloc[i]['Latitude'], sorted_working_data.iloc[i]['Longitude']],

      popup="Date: {}\nFatalities: {}".format(sorted_working_data.iloc[i]['Date'],str(int(sorted_working_data.iloc[i]['Fatalities']))),

      radius=(sorted_working_data.iloc[i]['Fatalities']+1)*500,

      color='crimson',

      fill=True,

      fill_color='crimson'

    ).add_to(interactive_map)

    



interactive_map        
IFrame('https://plotly.com/~Daedalus1/18/#/',width=800,height=600)

        