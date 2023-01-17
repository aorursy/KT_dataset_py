# Import required libraries
import pandas as pd
import numpy as np
from datetime import datetime
import folium
from folium import plugins
import urllib.request
import json
from collections import namedtuple
import sys
from math import log,tan,pi,radians
# Read the datasets
df_speed_radar_violations = pd.read_csv('../input/chicago-red-light-and-speed-camera-data/speed-camera-violations.csv')
df_red_light_radar_violations = pd.read_csv('../input/chicago-red-light-and-speed-camera-data/red-light-camera-violations.csv')
df_speed_radar_violations.head()
df_red_light_radar_violations.head()
def process_dataframes(df):
    df = df.dropna()
    # Aggregate violations by address
    agg_violations = pd.DataFrame(df.groupby(['ADDRESS', 'LATITUDE', 'LONGITUDE'])['VIOLATIONS'].sum()).reset_index()
    return agg_violations

def add_markers_folium_map(df, m, color):
    # Add markers in the location of the radars
    for i in range(0,len(df)):
        folium.Circle(
        location=[df['LATITUDE'][i], df['LONGITUDE'][i]],
        popup='ADDRESS: ' + df['ADDRESS'][i] + ', #VIOLATIONS = %i' %df['VIOLATIONS'][i],
        radius=float(df['VIOLATIONS'][i])*0.007,
        color=color,
        fill=True,
        fill_color=color
    ).add_to(m)
    return m
# Process dataframes for both speed and red light traffic violations
viol_speed = process_dataframes(df_speed_radar_violations)
viol_red_light = process_dataframes(df_red_light_radar_violations)

# Start the folium map with the location centered
m = folium.Map(location=[np.mean(viol_speed['LATITUDE']), np.mean(viol_speed['LONGITUDE'])])
 
# Add markers in the location of the radars
m = add_markers_folium_map(viol_speed, m, 'blue')
m = add_markers_folium_map(viol_red_light, m, 'red')
m
# Heat map
heat_viol = viol_speed[['LATITUDE', 'LONGITUDE']].values
m2 = folium.Map(location=[np.mean(viol_speed['LATITUDE']), np.mean(viol_speed['LONGITUDE'])])
m2.add_child(plugins.HeatMap(heat_viol, radius=25))
# Reference: https://github.com/jkgiesler/parse-chicago-neighborhoods
# This library is used to aggregate the violations by neighborhood in Chicago.
# THe function match latitude, longitude with the name of the neighborhood.
#globals
Pt = namedtuple('Pt','x,y')
Edge = namedtuple('Edge','a,b')
Poly = namedtuple('Poly','name,edges')
_eps = 1e-5
_huge = sys.float_info.max
_tiny = sys.float_info.min

def load_json():
    file_in = open("../input/chicago-neighborhoods-2012/Neighborhoods_2012_polygons.json")
    d = json.load(file_in)
    return d


def spherical_mercator_projection(longitude,latitude):
    #http://en.wikipedia.org/wiki/Mercator_projection <- invented in 1569!
    #http://stackoverflow.com/questions/4287780/detecting-whether-a-gps-coordinate-falls-within-a-polygon-on-a-map
    x = -longitude
    y = log(tan(radians(pi/4 + latitude/2)))
    return (x,y)


def rayintersectseg(p, edge):
    #http://rosettacode.org/wiki/Ray-casting_algorithm#Python
    #takes a point p=Pt() and an edge of two endpoints a,b=Pt() of a line segment returns boolean
    a,b = edge
    if a.y > b.y:
        a,b = b,a
    if p.y == a.y or p.y == b.y:
        p = Pt(p.x, p.y + _eps)
    intersect = False
 
    if (p.y > b.y or p.y < a.y) or (
        p.x > max(a.x, b.x)):
        return False
 
    if p.x < min(a.x, b.x):
        intersect = True
    else:
        if abs(a.x - b.x) > _tiny:
            m_red = (b.y - a.y) / float(b.x - a.x)
        else:
            m_red = _huge
        if abs(a.x - p.x) > _tiny:
            m_blue = (p.y - a.y) / float(p.x - a.x)
        else:
            m_blue = _huge
        
        intersect = m_blue >= m_red
    return intersect
 
def is_odd(x): 
    return x%2 == 1

def ispointinside(p, poly):
    ln = len(poly)
    return is_odd(sum(rayintersectseg(p, edge)
                    for edge in poly.edges ))

def get_all_neighborhoods():
    d = load_json()
    shape_list=[]
    for shape_idx in range(len(d['features'])):
        name = d['features'][shape_idx]['properties']['SEC_NEIGH']

        edges =[]
        for coordinate_idx in range(len(d['features'][shape_idx]['geometry']['coordinates'][0])-1):
            lon_1 = d['features'][shape_idx]['geometry']['coordinates'][0][coordinate_idx][0]
            lat_1 = d['features'][shape_idx]['geometry']['coordinates'][0][coordinate_idx][1]
            
            lon_2 = d['features'][shape_idx]['geometry']['coordinates'][0][coordinate_idx+1][0]
            lat_2 = d['features'][shape_idx]['geometry']['coordinates'][0][coordinate_idx+1][1]
        
            x1,y1 = spherical_mercator_projection(lon_1,lat_1)
            x2,y2 = spherical_mercator_projection(lon_2,lat_2)
            edges.append(Edge(a=Pt(x=x1,y=y1),b=Pt(x=x2,y=y2)))
        
        shape_list.append(Poly(name=name,edges=tuple(edges)))
    return shape_list

def find_neighborhood(test_long,test_lat,all_neighborhoods):
    x,y = spherical_mercator_projection(test_long,test_lat)
    for neighborhood in all_neighborhoods:
        correct_neighborhood = ispointinside(Pt(x=x,y=y),neighborhood)
        if correct_neighborhood:
            return neighborhood.name
all_neighborhoods = get_all_neighborhoods()
neighborhood = []
for i in range(len(viol_speed)):
    neighborhood.append(find_neighborhood(viol_speed['LONGITUDE'][i],
                                          viol_speed['LATITUDE'][i],all_neighborhoods))
viol_speed['NEIGHBORHOOD'] = pd.Series(neighborhood)

def process_neighborhoods(df):
    df = df.dropna()
    # Aggregate violations by address
    agg_neigh = pd.DataFrame(df.groupby(['NEIGHBORHOOD'])['VIOLATIONS'].sum()).reset_index()
    return agg_neigh
viol_speed_neighborhood = process_neighborhoods(viol_speed)
viol_speed_neighborhood.sort_values(by=['VIOLATIONS'], ascending = False).head(20)