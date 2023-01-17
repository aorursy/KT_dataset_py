import cv2

import json

import os

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd



%matplotlib inline



grid_name = '6010_4_2'

data_dir = '../input'

shape = (3345, 3396)



# Load grid CSV

grid_sizes = pd.read_csv(os.path.join(data_dir, 'grid_sizes.csv'), index_col=0)

grid_sizes.ix[grid_name]



def scale_coords(shape, grid_name, point):

    """Scale the coordinates of a polygon into the image coordinates for a grid cell"""

    w,h = shape

    Xmax, Ymin = grid_sizes.ix[grid_name][['Xmax', 'Ymin']]

    x,y = point[:,0], point[:,1]



    wp = float(w**2)/(w+1)

    xp = x/Xmax*wp



    hp = float(h**2)/(h+1)

    yp = y/Ymin*hp



    return np.concatenate([xp[:,None],yp[:,None]], axis=1)
# Load JSON of image overlays

sh_fname = os.path.join(data_dir, 'train_geojson_v3/%s/002_TR_L4_POOR_DIRT_CART_TRACK.geojson'%grid_name)

with open(sh_fname, 'r') as f:

    sh_json = json.load(f)



# Scale the polygon coordinates to match the pixels

polys = []

for sh in sh_json['features']:

    geom = np.array(sh['geometry']['coordinates'][0])

    geom_fixed = scale_coords(shape, grid_name, geom)



    pts = geom_fixed.astype(int)

    polys.append(pts)



# Create an empty mask and then fill in the polygons

mask = np.zeros(shape)

cv2.fillPoly(mask, polys, 1)

mask = mask.astype(bool)



plt.imshow(mask)