import folium 

import gpxpy

import os



baseDir = '../input/ptm strava data/PTM Strava Data/'



def overlayGPX(gpxData, zoom):

    '''

    overlay a gpx route on top of an OSM map using Folium

    some portions of this function were adapted

    from this post: https://stackoverflow.com/questions/54455657/

    how-can-i-plot-a-map-using-latitude-and-longitude-data-in-python-highlight-few

    '''

    gpx_file = open(gpxData, 'r')

    gpx = gpxpy.parse(gpx_file)

    points = []

    for track in gpx.tracks:

        for segment in track.segments:        

            for point in segment.points:

                points.append(tuple([point.latitude, point.longitude]))

    latitude = sum(p[0] for p in points)/len(points)

    longitude = sum(p[1] for p in points)/len(points)

    myMap = folium.Map(location=[latitude,longitude],zoom_start=zoom)

    folium.PolyLine(points, color="red", weight=2.5, opacity=1).add_to(myMap)

    return (myMap)
fileName = 'hike/Colorado_Longs_Peak_and_Chasm_Lake_Hike.gpx'

filePath = os.path.join(baseDir,fileName)

overlayGPX(filePath,14)
fileName = 'hike/Colorado_Belford_Oxford_and_Missouri_Mountains_Hike.gpx'

filePath = os.path.join(baseDir,fileName)

overlayGPX(filePath, 14)
fileName = 'hike/Berenthanti_Ghorepani_Ghandruk_Loop_Hike_Day_2_of_3_.gpx'

filePath = os.path.join(baseDir,fileName)

overlayGPX(filePath, 12)