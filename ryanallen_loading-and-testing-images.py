#Load libraries



%matplotlib inline



import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import shapely.wkt as wkt

import shapely.affinity

import cv2

import csv

import sys

import tifffile as tiff



from matplotlib.patches import Polygon as pltPoly

from collections import defaultdict

from shapely.geometry import MultiPolygon, Polygon
csv.field_size_limit(sys.maxsize) #set field size to max to avoid error later
df = pd.read_csv('../input/train_wkt_v2.csv')
df['ImageId'].unique() #what images are we working with?
polylist = {}

img = df[df.ImageId == '6170_2_4']

for c in img['ClassType'].unique():

    print(c)

    polylist[c] = wkt.loads(img[img['ClassType']==c].MultipolygonWKT.values[0])
polylist
#get the different classes and stack them to see how the polygons come together

fig, ax = plt.subplots(figsize=(10, 10))



for p in polylist:

    for polygon in polylist[p]:

        mpl_poly = pltPoly(np.array(polygon.exterior), color=plt.cm.Set1(p*10), lw=0, alpha=0.3)

        ax.add_patch(mpl_poly)



ax.relim()

ax.autoscale_view()
IMG_ID = '6170_2_4'

CLASS = '5' #Let's look at some trees
# Load grid size

x_max = y_min = None

for _im_id, _x, _y in csv.reader(open('../input/grid_sizes.csv')):

    if _im_id == IMG_ID:

        x_max, y_min = float(_x), float(_y)

        break



# Load train poly with shapely

train_polygons = None

for _im_id, _poly_type, _poly in csv.reader(open('../input/train_wkt_v4.csv')):

    if _im_id == IMG_ID and _poly_type == CLASS:

        train_polygons = shapely.wkt.loads(_poly)

        break



# Read image with tiff

im_rgb = tiff.imread('../input/three_band/{}.tif'.format(IMG_ID)).transpose([1, 2, 0])

im_size = im_rgb.shape[:2]
def get_scalers():

    h, w = im_size  # they are flipped so that mask_for_polygons works correctly

    w_ = w * (w / (w + 1))

    h_ = h * (h / (h + 1))

    return w_ / x_max, h_ / y_min



x_scaler, y_scaler = get_scalers()



train_polygons_scaled = shapely.affinity.scale(

    train_polygons, xfact=x_scaler, yfact=y_scaler, origin=(0, 0, 0))
def mask_for_polygons(polygons):

    img_mask = np.zeros(im_size, np.uint8)

    if not polygons:

        return img_mask

    int_coords = lambda x: np.array(x).round().astype(np.int32)

    exteriors = [int_coords(poly.exterior.coords) for poly in polygons]

    interiors = [int_coords(pi.coords) for poly in polygons

                 for pi in poly.interiors]

    cv2.fillPoly(img_mask, exteriors, 1)

    cv2.fillPoly(img_mask, interiors, 0)

    return img_mask



train_mask = mask_for_polygons(train_polygons_scaled)
def scale_percentile(matrix):

    w, h, d = matrix.shape

    matrix = np.reshape(matrix, [w * h, d]).astype(np.float64)

    # Get 2nd and 98th percentile

    mins = np.percentile(matrix, 1, axis=0)

    maxs = np.percentile(matrix, 99, axis=0) - mins

    matrix = (matrix - mins[None, :]) / maxs[None, :]

    matrix = np.reshape(matrix, [w, h, d])

    matrix = matrix.clip(0, 1)

    return matrix
tiff.imshow(255 * scale_percentile(im_rgb[2900:3200,2000:2300]))
#Show a black/white image of the selected feature type

def show_mask(m):

    # hack for nice display

    tiff.imshow(255 * np.stack([m, m, m]));

show_mask(train_mask[2900:3200,2000:2300])