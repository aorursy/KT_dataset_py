# import folium package 

import folium
a = folium.Map(

    location=[23.8859, 45.0792],

    tiles='Stamen Toner',

    zoom_start=6

)

# folium CircleMarker

folium.CircleMarker(

    # radius of CircleMarker

    radius=15,

    # Coordinates of location Masjid-al-Haram

    location=[21.4225, 39.8262],

    popup='Masjid-al-Haram',

    # color of CircleMarker

    color='crimson',

    fill=False,

).add_to(a)



folium.CircleMarker(

    # Coordinates of Masjid Nabwi

    location=[24.4672, 39.6111],

    # radius of CircleMarker

    radius=15,

    popup='',

    # color of CircleMarker

    color='#3186cc',

    fill=True,

    fill_color='#3186cc'

).add_to(a)



a
l = folium.Map(

    location=[20.5937, 78.9629],

    zoom_start=4.5,

    tiles='Stamen Terrain'

)



folium.Marker(

    location=[19.0760, 72.8777],

    popup=folium.Popup(max_width=450).add_child

).add_to(l)



folium.Marker(

    location=[28.7041, 77.1025],

    popup=folium.Popup(max_width=450).add_child

).add_to(l)



folium.Marker(

    location=[22.5726, 88.3639],

    popup=folium.Popup(max_width=450).add_child

).add_to(l)



folium.Marker(

    location=[13.0827, 80.2707],

    popup=folium.Popup(max_width=450).add_child

).add_to(l)



# Add line to map

folium.PolyLine(locations = [(19.0760, 72.8777), (28.7041, 77.1025), (22.5726, 88.3639), (13.0827, 80.2707), (19.0760, 72.8777)], 

                line_opacity = 0.5).add_to(l) 



l

# Kaaba View

m=folium.Map(

    location=[21.4225, 39.8262],

    zoom_start=16.5

)

m