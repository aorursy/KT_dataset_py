import os

import numpy as np

import tifffile

import json

from PIL import Image

import matplotlib

import matplotlib.pyplot as plt

import pandas as pd

from matplotlib.collections import PatchCollection



%matplotlib inline



data_dir = '../input'

grid_name = '6010_4_2'



# Load grid CSV

grid_sizes = pd.read_csv(os.path.join(data_dir, 'grid_sizes.csv'), index_col=0)

grid_sizes.ix[grid_name]

    

# Load JSON of image overlays

sh_fname = os.path.join(data_dir, 'train_geojson_v3/%s/002_TR_L4_POOR_DIRT_CART_TRACK.geojson'%grid_name)

with open(sh_fname, 'r') as f:

    sh_json = json.load(f)

    

# Load the tif file

im_fname = os.path.join(data_dir, 'three_band','%s.tif'%grid_name)

tif_data = tifffile.imread(im_fname).transpose([1,2,0])
def scale_coords(tif_data, grid_name, point):

    """Scale the coordinates of a polygon into the image coordinates for a grid cell"""

    w,h,_ = tif_data.shape

    Xmax, Ymin = grid_sizes.ix[grid_name][['Xmax', 'Ymin']]

    x,y = point[:,0], point[:,1]



    wp = float(w**2)/(w+1)

    xp = x/Xmax*wp



    hp = float(h**2)/(h+1)

    yp = y/Ymin*hp



    return np.concatenate([xp[:,None],yp[:,None]], axis=1)



def scale_percentile(matrix):

    """Fixes the pixel value range to 2%-98% original distribution of values"""

    orig_shape = matrix.shape

    matrix = np.reshape(matrix, [matrix.shape[0]*matrix.shape[1], 3]).astype(float)

    

    # Get 2nd and 98th percentile

    mins = np.percentile(matrix, 1, axis=0)

    maxs = np.percentile(matrix, 99, axis=0) - mins

    

    matrix = (matrix - mins[None,:])/maxs[None,:]

    matrix = np.reshape(matrix, orig_shape)

    matrix = matrix.clip(0,1)

    return matrix
# Show the image with the values scaled from 2-98 percentile to make them visible

fixed_im = scale_percentile(tif_data)

plt.imshow(fixed_im)
# Load JSON of image overlays, and convert it into image coordinates

def load_overlays(tile_name):

    """Get all of the polygon overlays for a tile.

    Returns a dict: {LABEL: POLYGON}"""

    dirname = os.path.join(data_dir, 'train_geojson_v3/%s/'%tile_name)

    fnames = [os.path.join(dirname, fname) for fname in 

              os.listdir(dirname) 

              if fname.endswith('.geojson') and not fname.startswith('Grid')]

    

    overlays = dict()

    for fname in fnames:

        with open(fname, 'r') as f:

            sh_json = json.load(f)

        label = sh_json['features'][0]['properties']['LABEL']

        print(label)

        

        polygons = []

        for sh in sh_json['features']:

            pts = scale_coords(tif_data, grid_name, np.array(sh['geometry']['coordinates'][0])).squeeze()

            

            # Remove badly formatted polygons

            if not ((len(pts.shape)==2) and (pts.shape[1]==2) and (pts.shape[0] > 2)):

                continue

            polygons.append(pts)

            

        overlays[label] = polygons



    return overlays

    

overlays_poly = load_overlays(grid_name)
# Put polygons on top of the image



fig, ax = plt.subplots(figsize=(6,6))

patches = []

for pts in overlays_poly['POOR_DIRT_CART_TRACK']:

    

    poly = matplotlib.patches.Polygon(pts)

    patches.append(poly)



p = PatchCollection(patches, cmap=matplotlib.cm.jet, alpha=0.4)



colors = 100*np.random.rand(len(patches))

p.set_array(np.array(colors))



ax.imshow(fixed_im)

ax.add_collection(p)



plt.show()