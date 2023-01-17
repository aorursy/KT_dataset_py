import numpy as np

import pandas as pd

import geopandas as gpd

from geopandas.tools import geocode

import math

from collections import namedtuple



import folium

from folium import Choropleth, Circle, Marker

from folium.plugins import HeatMap, MarkerCluster, TimestampedGeoJson



import datetime

import os
import os

for dirname, _, filenames in os.walk('../input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# loading the data

chinaVectors = "../input/china-regions-map/china.json"

df = pd.read_csv("../input/coronavirus-latlon-dataset/CV_LatLon_21Jan_12Mar.csv", index_col = 0)
# lets see how the big picture of virus propagation looked on 12th of March

df12OfMarch = df.loc[df.date == '3/12/20', 

                         ['state', 

                          'country', 

                          'confirmed', 

                          'recovered', 

                          'death', 

                          'lat', 

                          'lon']]
def radiusMinMaxer(radius):

    radiusMin = 2

    radiusMax = 40

    if radius != 0:

        if radius < radiusMin:

            radius = radiusMin

        if radius > radiusMax:

            radius = radiusMax

    return radius
# and now the fun part, getting it all on the map. I borrowed the style and some ideas from this article: https://towardsdatascience.com/visualizing-bike-mobility-in-london-using-interactive-maps-for-absolute-beginners-3b9f55ccb59

colorConfirmed = '#ffbf80'

colorRecovered = '#0A5E2AFF'

colorDead = '#E80018'

circleFillOpacity = 0.2



map = folium.Map(location=[15.632909, 14.911222], 

                 tiles = "CartoDB dark_matter",

                 detect_retina = True,

                 zoom_start=2)



# map layers

layerFlights = folium.FeatureGroup(name='<span style="color: black;">Flights</span>')

layerConfirmed = folium.FeatureGroup(name='<span style=\\"color: #EFEFE8FF;\\">Confirmed infected</span>')

layerDead = folium.FeatureGroup(name='<span style=\\"color: #E80018;\\">Dead</span>')

layerRecovered = folium.FeatureGroup(name='<span style=\\"color: #0A5E2AFF;\\">Recovered from virus</span>')  



# the choropleth idea togeather with circles was adviced by: https://www.kaggle.com/gpreda/tracking-the-spread-of-2019-coronavirus

folium.Choropleth(

                geo_data=chinaVectors,

                name='Choropleth',

                key_on='feature.properties.name',

                fill_color='yellow',

                fill_opacity=0.18,

                line_opacity=0.7

                ).add_to(map)



# coordinates of Huabei province, thats where first infected travelers were departuring from.

departurePoint = [df12OfMarch.loc[df12OfMarch.state == 'Hubei', 

                                  'lat'].values[0], df12OfMarch.loc[df12OfMarch.state == 'Hubei', 

                                                                    'lon'].values[0]]



for i, row in df12OfMarch.iterrows():

    lat = row.lat

    lon = row.lon

    country = row.country

    province = row.state

    recovered = row.recovered

    death = row.death

    confirmed = row.confirmed



    radiusConfirmed = radiusMinMaxer(np.sqrt(confirmed))

    radiusRecovered = radiusMinMaxer(np.sqrt(recovered))

    radiusDead = radiusMinMaxer(np.sqrt(death))

    

    # coordinates of infected travelers arrivals

    arrivalPoint = [lat, lon]



    if row.state != '0':

        popup = 'Flight:&nbsp;' + 'Hubei,&nbsp;China&nbsp;-&nbsp;' + row.state + ',&nbsp;' + row.country

    else:

        popup = 'Flight:&nbsp;' + 'Hubei,&nbsp;China&nbsp;-&nbsp;' + row.country

        

    folium.PolyLine(locations=[departurePoint, arrivalPoint], 

                      color='white', 

                      weight = 0.5,

                      opacity = 0.3,

                      popup = popup

                       ).add_to(layerFlights)



    popupConfirmed = str(country) + ' ' + str(province) + '(Confirmed='+str(row.confirmed) + ' Deaths=' + str(death) + ' Recovered=' + str(recovered) + ')'



    folium.CircleMarker(location = [lat,lon], 

                        radius = radiusConfirmed, 

                        popup = popupConfirmed, 

                        color = colorConfirmed, 

                        fill_opacity = 0.3,

                          weight = 1, 

                          fill = True, 

                          fillColor = colorConfirmed

                           ).add_to(layerConfirmed)

    

    if row.recovered != 0:

        popupRecovered = str(country) + ' ' + str(province) + '(Confirmed='+str(row.confirmed) + ' Deaths=' + str(death) + ' Recovered=' + str(recovered) + ')'



        folium.CircleMarker(location = [lat,lon], 

                            radius = radiusRecovered, 

                            popup = popupRecovered, 

                            color = colorRecovered, 

                            fill_opacity = circleFillOpacity,

                              weight = 1, 

                              fill = True, 

                              fillColor = colorRecovered

                               ).add_to(layerRecovered) 

        

    if row.death != 0:

        popupDead = str(country) + ' ' + str(province) + '(Confirmed='+str(row.confirmed) + ' Deaths=' + str(death) + ' Recovered=' + str(recovered) + ')'



        folium.CircleMarker(location = [lat,lon], 

                            radius = radiusDead, 

                            popup = popupDead, 

                            color = colorDead, 

                            fill_opacity = circleFillOpacity,

                              weight = 1, 

                              fill = True, 

                              fillColor = colorDead

                               ).add_to(layerRecovered) 



layerFlights.add_to(map)

layerConfirmed.add_to(map)

layerRecovered.add_to(map)

layerDead.add_to(map)



folium.map.LayerControl('bottomleft', collapsed=False).add_to(map)



map
map = folium.Map(location=[32.902807, 101.089332], 

                 tiles = "CartoDB dark_matter",

                 detect_retina = True,

                 zoom_start=4)



# map layers

layerFlights = folium.FeatureGroup(name='<span style="color: black;">Flights</span>')

layerConfirmed = folium.FeatureGroup(name='<span style=\\"color: #EFEFE8FF;\\">Confirmed infected</span>')

layerDead = folium.FeatureGroup(name='<span style=\\"color: #E80018;\\">Dead</span>')

layerRecovered = folium.FeatureGroup(name='<span style=\\"color: #0A5E2AFF;\\">Recovered from virus</span>')  



# the choropleth idea togeather with circles was adviced by: https://www.kaggle.com/gpreda/tracking-the-spread-of-2019-coronavirus

folium.Choropleth(

                geo_data=chinaVectors,

                name='Choropleth',

                key_on='feature.properties.name',

                fill_color='yellow',

                fill_opacity=0.18,

                line_opacity=0.7

                ).add_to(map)







for i, row in df12OfMarch.iterrows():

    lat = row.lat

    lon = row.lon

    country = row.country

    province = row.state

    recovered = row.recovered

    death = row.death

    confirmed = row.confirmed



    radiusConfirmed = radiusMinMaxer(np.sqrt(confirmed))

    radiusRecovered = radiusMinMaxer(np.sqrt(recovered))

    radiusDead = radiusMinMaxer(np.sqrt(death))

    

    # coordinates of infected travelers arrivals

    arrivalPoint = [lat, lon]



    if row.state != '0':

        popup = 'Flight:&nbsp;' + 'Hubei,&nbsp;China&nbsp;-&nbsp;' + row.state + ',&nbsp;' + row.country

    else:

        popup = 'Flight:&nbsp;' + 'Hubei,&nbsp;China&nbsp;-&nbsp;' + row.country

        

    folium.PolyLine(locations=[departurePoint, arrivalPoint], 

                      color='white', 

                      weight = 0.5,

                      opacity = 0.3,

                      popup = popup

                       ).add_to(layerFlights)



    popupConfirmed = str(country) + ' ' + str(province) + '(Confirmed='+str(row.confirmed) + ' Deaths=' + str(death) + ' Recovered=' + str(recovered) + ')'



    folium.CircleMarker(location = [lat,lon], 

                        radius = radiusConfirmed, 

                        popup = popupConfirmed, 

                        color = colorConfirmed, 

                        fill_opacity = 0.3,

                          weight = 1, 

                          fill = True, 

                          fillColor = colorConfirmed

                           ).add_to(layerConfirmed)

    

    if row.recovered != 0:

        popupRecovered = str(country) + ' ' + str(province) + '(Confirmed='+str(row.confirmed) + ' Deaths=' + str(death) + ' Recovered=' + str(recovered) + ')'



        folium.CircleMarker(location = [lat,lon], 

                            radius = radiusRecovered, 

                            popup = popupRecovered, 

                            color = colorRecovered, 

                            fill_opacity = circleFillOpacity,

                              weight = 1, 

                              fill = True, 

                              fillColor = colorRecovered

                               ).add_to(layerRecovered) 

        

    if row.death != 0:

        popupDead = str(country) + ' ' + str(province) + '(Confirmed='+str(row.confirmed) + ' Deaths=' + str(death) + ' Recovered=' + str(recovered) + ')'



        folium.CircleMarker(location = [lat,lon], 

                            radius = radiusDead, 

                            popup = popupDead, 

                            color = colorDead, 

                            fill_opacity = circleFillOpacity,

                              weight = 1, 

                              fill = True, 

                              fillColor = colorDead

                               ).add_to(layerRecovered) 



layerFlights.add_to(map)

layerConfirmed.add_to(map)

layerRecovered.add_to(map)

layerDead.add_to(map)



folium.map.LayerControl('bottomleft', collapsed=False).add_to(map)



map
radiusMin = 2

radiusMax = 50

colorConfirmed = '#E80018'

colorRecovered = '#81D8D0'



for date in df.date.unique():

    print('date=', date)

    _df = df.loc[df.date == date, 

                ['state', 

                'country', 

                'confirmed', 

                'recovered', 

                'death', 

                'lat', 

                'lon']]

    _df.reset_index(drop = True, inplace = True)

    _map = folium.Map(location=[15.632909, 14.911222], 

                 tiles = "CartoDB dark_matter", 

                 zoom_start=2)



    folium.Choropleth(

                        geo_data=chinaVectors,

                        name='choropleth',

                        key_on='feature.properties.name',

                        fill_color='yellow',

                        fill_opacity=0.18,

                        line_opacity=0.7).add_to(_map)



    for i, row in _df.iterrows():

        if row.confirmed != 0:

            arrivalPoint = [row.lat, row.lon]



            if row.state != '0':

                popup = 'Flight:&nbsp;' + 'Hubei,&nbsp;China&nbsp;-&nbsp;' + row['state'] + ',&nbsp;' + row['country']

            else:

                popup = 'Flight:&nbsp;' + 'Hubei,&nbsp;China&nbsp;-&nbsp;' + row['country']

                folium.PolyLine(locations=[departurePoint, arrivalPoint], 

                              color='white', 

                              weight = 0.5,

                              opacity = 0.4,

                              popup = popup).add_to(_map)



        lat = row.lat

        lon = row.lon

        country = row.country

        province = row.state

        recovered = row.recovered

        death = row.death

        confirmed = row.confirmed



        radiusConfirmed = radiusMinMaxer(np.sqrt(confirmed))



        popupConfirmed = str(country) + ' ' + str(province) + '(Confirmed='+str(row.confirmed) + ' Deaths=' + str(death) + ' Recovered=' + str(recovered) + ')'

        folium.CircleMarker(location = [lat,lon], 

                            radius = radiusConfirmed, 

                            popup = popupConfirmed, 

                            color = colorConfirmed, 

                            fill_opacity = 0.2,

                            weight = 1, 

                            fill = True, 

                            fillColor = colorConfirmed).add_to(_map)

      

        if row.recovered != 0:

            radiusRecovered = radiusMinMaxer(np.sqrt(recovered))



        popupRecovered = str(country) + ' ' + str(province) + '(Confirmed='+str(row.confirmed) + ' Deaths=' + str(death) + ' Recovered=' + str(recovered) + ')'



        folium.CircleMarker(location = [lat,lon], 

                              radius = radiusRecovered, 

                              popup = popupRecovered, 

                              color = colorRecovered, 

                              fill_opacity = 0.2,

                              weight = 1, 

                              fill = True, 

                              fillColor = colorRecovered).add_to(_map) 



    path = '/kaggle/working/'

    f = path + 'map' + str(date).replace('/', '-') + '.html'

    _map.save(f)

dfConfirmed = df.loc[df.confirmed != 0, 

                         ['state', 

                          'country', 

                          'confirmed', 

                          'lat', 

                          'lon',

                          'date']]



dfRecovered = df.loc[df.recovered != 0, 

                         ['state', 

                          'country', 

                          'recovered', 

                          'lat', 

                          'lon',

                         'date']]



dfDead = df.loc[df.death != 0, 

                         ['state', 

                          'country', 

                          'death', 

                          'lat', 

                          'lon',

                         'date']]
def create_geojson_features(dfConfirmed,

                            dfRecovered, 

                            dfDead,

                            radiusMax = 40, 

                            radiusMin = 2, 

                            colorConfirmed = colorConfirmed,

                            colorRecovered = colorRecovered,

                            colorDead = colorDead,

                            weight = 1,

                            fillOpacity = 0.2

                            ):



    print('> Creating GeoJSON features...')

    features = []

    feature = []

    

    for _, row in dfConfirmed.iterrows():

        radius = np.sqrt(row.confirmed)

        if radius != 0:

          if radius < radiusMin:

            radius = radiusMin



          if radius > radiusMax:

            radius = radiusMax



          feature = {

              'type': 'Feature',

              'geometry': {

                  'type':'Point', 

                  'coordinates':[row.lon, row.lat]

              },

              'properties': {

                  'time': row.date.__str__(),

                  'style': {'color' : colorConfirmed},

                  'icon': 'circle',

                  'iconstyle':{

                      'fillColor': colorConfirmed,

                      'fillOpacity': fillOpacity,

                      'stroke': 'true',

                      'radius': radius,

                      'weight': weight

                  }

              }

        }

        features.append(feature)



    for _, row in dfDead.iterrows():

        radius = np.sqrt(row.death)

        if radius != 0:

          if radius < radiusMin:

            radius = radiusMin



          if radius > radiusMax:

            radius = radiusMax

          popup = str(row.country) + ' ' + str(row.state) + '(Deaths=' + str(row.death) +')'

          feature = {

              'type': 'Feature',

              'geometry': {

                  'type':'Point', 

                  'coordinates':[row.lon,row.lat]

              },

              'properties': {

                  'time': row.date.__str__(),

                  'style': {'color' : colorDead},

                  'icon': 'circle',

                  'iconstyle':{

                      'fillColor': colorDead,

                      'fillOpacity': fillOpacity,

                      'stroke': 'true',

                      'radius': radius,

                      'weight': weight,

                      'popup': popup

                  }

              }

        }

        features.append(feature)



    for _, row in dfRecovered.iterrows():

        radius = np.sqrt(row.recovered)

        if radius != 0:

          if radius < radiusMin:

            radius = radiusMin



          if radius > radiusMax:

            radius = radiusMax



          feature = {

              'type': 'Feature',

              'geometry': {

                  'type':'Point', 

                  'coordinates':[row.lon,row.lat]

              },

              'properties': {

                  'time': row.date.__str__(),

                  'style': {'color' : colorRecovered},

                  'icon': 'circle',

                  'iconstyle':{

                      'fillColor': colorRecovered,

                      'fillOpacity': fillOpacity,

                      'stroke': 'true',

                      'radius': radius,

                      'weight': weight

                  }

              }

        }

        features.append(feature)

    

    

    return features
def make_map(features, caption):

    print('> Making map...')

    coordinates=[15.632909, 14.911222]

    map = folium.Map(location=coordinates, 

                               control_scale=True, 

                               zoom_start=2,

                               tiles = 'CartoDB dark_matter',

                               detect_retina = True

                              )

    

    folium.Choropleth(

        geo_data=chinaVectors,

        name='Choropleth',

        key_on='feature.properties.name',

        fill_color='yellow',

        fill_opacity=0.18,

        line_opacity=0.7

        ).add_to(map)





    TimestampedGeoJson(

        {'type': 'FeatureCollection',

        'features': features}

        , period='P1D'

        , duration='P1D'

        , add_last_point=True

        , auto_play=False

        , loop=False

        , max_speed=1

        , loop_button=True

        , date_options='YYYY/MM/DD'

        , time_slider_drag_update=True

        , transition_time = 500

    ).add_to(map)

    

    map.caption = caption

    print('> Done.')

    return map
features = create_geojson_features(dfConfirmed, dfRecovered, dfDead, fillOpacity=0.3, weight = 1)

make_map(features, caption = "Coronavirus propagation 21Jan–13March, 2020.")