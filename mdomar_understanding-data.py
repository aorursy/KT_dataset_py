# Summarized several notebooks of other Kagglers. Big Thanks to those guys

# shawn:

# https://www.kaggle.com/shawn775/dstl-satellite-imagery-feature-detection/polygon-transformation-to-match-image/comments

# Oleg Medvedev:

# https://www.kaggle.com/torrinos/dstl-satellite-imagery-feature-detection/exploration-and-plotting
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os        

import tifffile as tiff  #For the images

import cv2    #for better image processing

from shapely.wkt import loads as wkt_loads

from shapely import affinity

from matplotlib.patches import Polygon

import matplotlib.pyplot as plt

import matplotlib.image as mpimg





df = pd.read_csv('../input/train_wkt_v3.csv')

gs = pd.read_csv('../input/grid_sizes.csv', names=['ImageId', 'Xmax', 'Ymin'], skiprows=1)

print(df['ImageId'].unique())
# Class Type is Class of Objects:

# 1. Buildings - large building, residential, non-residential, fuel storage facility, fortified building

# 2. Misc. Manmade structures 

# 3. Road 

# 4. Track - poor/dirt/cart track, footpath/trail

# 5. Trees - woodland, hedgerows, groups of trees, standalone trees

# 6. Crops - contour ploughing/cropland, grain (wheat) crops, row (potatoes, turnips) crops

# 7. Waterway 

# 8. Standing water

# 9. Vehicle Large - large vehicle (e.g. lorry, truck,bus), logistics vehicle

# 10. Vehicle Small - small vehicle (car, van), motorbike
image_id = '6120_2_2'

filename = os.path.join('..', 'input', 'three_band', '{}.tif'.format(image_id))



img = tiff.imread(filename)   

tiff.imshow(img)
filename_A = os.path.join('..', 'input', 'sixteen_band/{}_A.tif'.format(image_id))

img_A = tiff.imread(filename_A)   

tiff.imshow(img_A)
filename_M = os.path.join('..', 'input', 'sixteen_band/{}_M.tif'.format(image_id))

img_M = tiff.imread(filename_M)   

tiff.imshow(img_M)
filename_P = os.path.join('..', 'input', 'sixteen_band/{}_P.tif'.format(image_id))

img_P = tiff.imread(filename_P)   

tiff.imshow(img_P)
polygonsList ={}

image = df[df.ImageId == image_id]

for cType in image.ClassType.unique():

    polygonsList[cType] = wkt_loads(image[image.ClassType == cType].MultipolygonWKT.values[0])

    

# plot using matplotlib

fig, ax = plt.subplots(figsize=(8, 8))



# plotting, color by class type

for p in polygonsList:

    for polygon in polygonsList[p]:

        mpl_poly = Polygon(np.array(polygon.exterior), color=plt.cm.Set1(p*10), lw=0, alpha=0.3)

        ax.add_patch(mpl_poly)



ax.relim()

ax.autoscale_view()
ax
polygonsList ={}

image = df[df.ImageId == image_id]

for cType in image.ClassType.unique():

    polygonsList[cType] = wkt_loads(image[image.ClassType == cType].MultipolygonWKT.values[0])
# plot using matplotlib

fig, ax = plt.subplots(figsize=(8, 8))



# plotting, color by class type

for p in polygonsList:

    for polygon in polygonsList[p]:

        mpl_poly = Polygon(np.array(polygon.exterior), color=plt.cm.Set1(p*10), lw=0, alpha=0.3)

        ax.add_patch(mpl_poly)



ax.relim()

ax.autoscale_view()
np.array(mpl_poly)