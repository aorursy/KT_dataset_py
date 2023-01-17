import pandas as pd

import numpy as np



from shapely.wkt import loads

from shapely import affinity

from matplotlib.patches import Polygon

import matplotlib.pyplot as plt



import tifffile as tiff



import glob, os

df = pd.read_csv('../input/train_wkt_v3.csv')

df.head()
gs = pd.read_csv('../input/grid_sizes.csv', names=['ImageId', 'Xmax', 'Ymin'], skiprows=1)

print(gs.head())
# First Image

polygonsList ={}

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
import tifffile as tiff



img_filename_6100_1_3 = '../input/three_band/6100_1_3.tif'



image = tiff.imread(img_filename_6100_1_3) 

tiff.imshow(image)
# Dimensions

dims = np.shape(image)

print(dims)
# RGB values

np.min(image), np.max(image)
# For any image specific classification, clustering, etc. transforms we'll want to 

# collapse spatial dimensions so that we have a matrix of pixels by color channels.



pixel_matrix = np.reshape(image, (dims[1] * dims[2] , dims[0]))

print(np.shape(pixel_matrix))
# Scatter plots are a go to to look for clusters and separatbility in the data, 

# but these are busy and don't reveal density well, so we switch to using 2d histograms instead. 

# The data between bands is really correlated, typical with visible imagery and

# why most satellite image analysts prefer to at least have near infrared values.





#plt.scatter(pixel_matrix[:,0], pixel_matrix[:,1])

_ = plt.hist2d(pixel_matrix[:,1], pixel_matrix[:,2], bins=(50,50))
img_filename_6100_1_4 = '../input/three_band/6110_1_4.tif'



image4 = tiff.imread(img_filename_6100_1_4) 

dims = np.shape(image4)

pixel_matrix4 = np.reshape(image4, (dims[1] * dims[2], dims[0]))

_ = plt.hist2d(pixel_matrix4[:,1], pixel_matrix4[:,2], bins=(50,50))

# for variations between the images:

#_ = plt.hist2d(pixel_matrix[:,2], pixel_matrix4[:,2], bins=(50,50))

# Rudimentary Transforms, Edge Detection, Texture



import skimage

from skimage.feature import greycomatrix, greycoprops

from skimage.filters import sobel
# SOBEL EDGE DETECTION

# A Sobel filter is one means of getting a basic edge magnitude/gradient image.

# Can be useful to threshold and find prominent linear features, etc. 

# Several other similar filters in skimage.filters are also good

# edge detectors: roberts, scharr, etc. and you can control direction, i.e. use an anisotropic version.



# a sobel filter is a basic way to get an edge magnitude/gradient image

#fig = plt.figure(figsize=(8, 8))

#plt.imshow(sobel(image[:750,:750,2]))

tiff.imshow(sobel(image[:750,:750,2]))