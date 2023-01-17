# Creating maps using folium libraries



# importing libraries



import folium

import pandas
# importing dataset



data = pandas.read_csv("../input/volcanoes.txt")
vname = list(data['Name'])

lat = list(data['Latitudes'])

lon = list(data['Longitudes'])

elev = list(data['Elevation'])

status = list(data['Status'])
map = folium.Map(location=[36.2048,138.2529],zoom_start = 6,tiles="Mapbox Bright")


def color_producer(stt):

    if stt == "Active":

        return 'red'

    else:

        return 'yellow'
fgv = folium.FeatureGroup(name="volcanoes")
for lt, ln, el, st, vnm in zip(lat,lon,elev,status,vname):

    fgv.add_child(folium.CircleMarker(location=[lt,ln],radius = 5

                                      ,popup = vnm + "," + str(el)+" meters" + "," +st,

                                      fill_color = color_producer(st), fill=True, color = 'grey', fill_opacity=1))

map.add_child(fgv)
map.add_child(folium.LayerControl())



map.save("Japan-Volcanoes.html")