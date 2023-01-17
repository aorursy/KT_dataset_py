import folium
import pandas as pd
dataset = pd.read_excel('../input/dataset/waterwatch_clean2 (1).xlsx')
dataset.info()
dataset.columns
dataset.head(10)
cpmap = folium.Map(location=[-33.967,18.485], zoom_start=10)
dataset.columns
dataset['Oct 2017\nkl/month'][0]
dataset['Suburb'][0]
place = dataset[ ['Latitude', 'Longitude'] ]
type(place)
# dataframe : numpy array : python list

place = place.values.tolist()
type(place)
def map_function(i,color):

    folium.Marker(

            location=point, 

            popup=dataset['Suburb'][i],

            icon=folium.Icon(color=color, icon='cloud', icon_color='white' )

        ).add_to(cpmap)
i=0

for point in place:                               

    if dataset['Oct 2017\nkl/month'][i] >= 10:  #i.e. red where Oct 2017\nkl/month>=10 else green

        map_function(i,'red')

    else:

        map_function(i,'green')

    i+=1
cpmap
cpmap.save('mymap.html') 





#saving the map in html file
#to add circle to the areas

#help to overcome congestion of markers





m = folium.Map(

    location=[45.5236, -122.6750],

    tiles='Stamen Toner',

    zoom_start=13

)



folium.Circle(

    radius=200,

    location=[45.5244, -122.6699],

    popup='The Waterfront',

    color='crimson',

    fill=False,

).add_to(m)



folium.CircleMarker(

    location=[45.5215, -122.6261],

    radius=50,

    popup='Laurelhurst Park',

    color='#3186cc',

    fill=True,

    fill_color='#3186cc'

).add_to(m)





m