import folium

from folium.features import Choropleth

from shapely.geometry import Polygon, Point, mapping

import json
# Hats off to folium/leaflet for making this necessary, as the Milanese neighbourhoods plotted using the choropleth otherwise will end up quite literally in Somalia.

def swap_geojson_coordinates(original_filename, new_filename):

    # We will open the original file

    with open(original_filename, 'r') as f:

        data = json.load(f)

    #going through each "layer"

    for feature in data['features']:

        coords = feature['geometry']['coordinates']

        #coordList is a list of coordinates identifying each polygon

        for coordList in coords:

            #each point, expressed as a latitude, longitude pair

            for coordPair in coordList:

                coordPair[0],coordPair[1] = coordPair[1], coordPair[0]

    # here is the new file

    with open(new_filename, 'w') as f:

        json.dump(data, f)

        

def create_neighbourhoods_dictionary(filename):

    with open(filename) as f:

        neighbourhood_dictionary = {}

        js = json.load(f)

        for feature in js['features']:

            coordinates = [(l[0], l[1]) for l in feature['geometry']['coordinates'][0]]

            neighbourhood_dictionary[feature['properties']['NIL']] = coordinates

    return neighbourhood_dictionary

    

def create_neighbourhood_centres_dictionary(neigh_dict):

    neigh_centres_dict = {}

    for key, value in neigh_dict.items():

        neigh_centres_dict[key] = mapping(Polygon(value).representative_point())['coordinates']

    return neigh_centres_dict





# We swap the coordinates in order to get them to be in Milan according to the latitude-longitude convention.

# if on Kaggle swap_geojson_coordinates('../input/nilzone.geojson', 'nilzone_swapped.geojson')

swap_geojson_coordinates('../input/nilzone.geojson', 'nilzone_swapped.geojson')

NIL_coordinates = create_neighbourhoods_dictionary('nilzone_swapped.geojson')  

neighbourhoods_centres = create_neighbourhood_centres_dictionary(NIL_coordinates)
neighbourhoods_map = folium.Map(location=[45.464211, 9.191383], tiles="cartodbdark_matter", zoom_start=13)

for key, value in neighbourhoods_centres.items():

    popup = str(key)

    folium.Marker([value[0], value[1]], popup=popup).add_to(neighbourhoods_map)



# We have to use the "longitude-latitude" coordinates, that is, the original ones, otherwise off to Somalia they go.

Choropleth(geo_data='../input/nilzone.geojson', fill_color='gray', line_color='green', fill_opacity=0.4,

            line_weight=3).add_to(neighbourhoods_map)



neighbourhoods_map.save(outfile= "milanese_neighbourhoods.html")

neighbourhoods_map