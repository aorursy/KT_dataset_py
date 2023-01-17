import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import json



with open('../input/gbdouble.json') as data_file:    

    geojson = json.load(data_file)



print('One Entry:: ', geojson['features'][0])

allCoordinates = []

for geo in geojson['features']:

    if geo['geometry']:

        thisPiece =  geo['geometry']['coordinates'] 

        #print(thisPiece)

        #if len(allCoordinates) > 100:

        #    break

            

        if type(thisPiece[0][0]) == float:

            thisPiece = [ (el[1], el[0]) for el in thisPiece ]

        else:

            thisPiece2 = []

            for t2 in thisPiece:

                for t3 in t2:

                    thisPiece2.append( (t3[1], t3[0]) )

            thisPiece = thisPiece2

        allCoordinates.append(thisPiece)
import folium

%matplotlib inline

map_ = folium.Map(location=[45.5017, -73.5673], zoom_start=11)



for idx, piece in enumerate(allCoordinates):

    #print(idx)

    if idx > 10000:

        break

    #print(piece)

    folium.PolyLine(piece, color="red", weight=2.5, opacity=1).add_to(map_)



    

#map_



#map_.create_map(path='streetsides.html')