import pandas as pd

import numpy as np

from shapely.wkt import loads

from matplotlib.patches import Polygon

import matplotlib.pyplot as plt



df = pd.read_csv('../input/train_wkt_v2.csv')

df.head()
# Use just first image

polygonsList = {}

image = df[df.ImageId == '6100_1_3']

for cType in image.ClassType.unique():

    polygonsList[cType] = loads(image[image.ClassType == cType].MultipolygonWKT.values[0])
# plot using matplotlib

fig, ax = plt.subplots(figsize=(8, 8))



# plotting, color by class type

for p in polygonsList:

    for polygon in polygonsList[p]:

        mpl_poly = Polygon(np.array(polygon.exterior), color=plt.cm.Set1(p*10), lw=0, alpha=0.3)

        ax.add_patch(mpl_poly)



ax.relim()

ax.autoscale_view()
# number of objects on the image by type

'''

1. Buildings

2. Misc. Manmade structures 

3. Road 

4. Track - poor/dirt/cart track, footpath/trail

5. Trees - woodland, hedgerows, groups of trees, standalone trees

6. Crops - contour ploughing/cropland, grain (wheat) crops, row (potatoes, turnips) crops

7. Waterway 

8. Standing water

9. Vehicle Large - large vehicle (e.g. lorry, truck,bus), logistics vehicle

10. Vehicle Small - small vehicle (car, van), motorbike

'''

for p in polygonsList:

    print("Type: {:4d}, objects: {}".format(p,len(polygonsList[p].geoms)))
# number of images in available kernel dataset?

df.ImageId.unique()
# convert to shapely, get geometries and pivot

df['polygons'] = df.apply(lambda row: loads(row.MultipolygonWKT),axis=1)

df['nPolygons'] = df.apply(lambda row: len(row['polygons'].geoms),axis=1)



pvt = df.pivot(index='ImageId', columns='ClassType', values='nPolygons')

pvt
fig, ax = plt.subplots(figsize=(10, 4))

ax.set_aspect('equal')

plt.imshow(pvt.T, interpolation='nearest', cmap=plt.cm.Blues, extent=[0,22,10,1])

plt.yticks(np.arange(1, 11, 1.0))

plt.title('Number of objects by type')

plt.ylabel('Class Type')

plt.xlabel('Image')

plt.colorbar()

plt.show()
from scipy.stats import pearsonr

print("Trees vs Buildings: {:5.4f}".format(pearsonr(pvt[1],pvt[5])[0]))

print("Trees vs Buildings and Structures: {:5.4f}".format(pearsonr(pvt[1]+pvt[2],pvt[5])[0]))
for im in df.ImageId.unique():

    image = df[df.ImageId == im]

    for cType in image.ClassType.unique():

        polygonsList[cType] = loads(image[image.ClassType == cType].MultipolygonWKT.values[0])

    

    # plot using matplotlib

    fig, ax = plt.subplots(figsize=(8, 8))



    # plotting, color by class type

    for p in polygonsList:

        for polygon in polygonsList[p]:

            mpl_poly = Polygon(np.array(polygon.exterior), color=plt.cm.Set1(p*10), lw=0, alpha=0.3)

            ax.add_patch(mpl_poly)



    ax.relim()

    ax.autoscale_view()

    plt.show()