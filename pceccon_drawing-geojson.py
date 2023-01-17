import numpy as np 

import pandas as pd

import geojson

import matplotlib.pyplot as plt



from descartes import PolygonPatch

from mpl_toolkits.basemap import Basemap



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

json_data = geojson.load(open('../input/community-districts-polygon.geojson'))

json_data.keys()
polygons = json_data['features']

len(polygons)
polygons[10]['geometry']['coordinates']
west, south, east, north = -74.26, 40.50, -73.70, 40.92

BLUE = '#6699cc'



fig = plt.figure(figsize=(14,8))

ax = fig.gca() 



for i in range(len(polygons)):

    coordlist = polygons[i]['geometry']['coordinates']

    poly = {'type':'Polygon', 'coordinates':coordlist}

    ax.add_patch(PolygonPatch(poly, fc=BLUE, ec=BLUE, alpha=0.5, zorder=2))

    ax.axis('scaled')

plt.draw()

plt.show()