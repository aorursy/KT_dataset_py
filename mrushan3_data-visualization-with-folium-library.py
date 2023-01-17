# import folium package 

import folium

# Here we pass coordinates of India 

# and starting Zoom level = 4

folium.Map(

    location=[20.5937,78.9629],

    zoom_start=4

)
a = folium.Map(

    location=[20.5937, 78.9629],

    tiles='Stamen Toner',

    zoom_start=6

)

# folium CircleMarker

folium.CircleMarker(

    # radius of CircleMarker

    radius=15,

    # Coordinates of location Ahmedabad

    location=[23.0225, 72.5714],

    popup='Ahmedabad',

    # color of CircleMarker

    color='crimson',

    fill=False,

).add_to(a)



folium.CircleMarker(

    # Coordinates of location Surat

    location=[21.1702, 72.8311],

    # radius of CircleMarker

    radius=15,

    popup='Surat',

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
