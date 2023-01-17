# Summarized several notebooks of other Kagglers. Big Thanks to those guys

# shawn:

# https://www.kaggle.com/shawn775/dstl-satellite-imagery-feature-detection/polygon-transformation-to-match-image/comments

# Oleg Medvedev:

# https://www.kaggle.com/torrinos/dstl-satellite-imagery-feature-detection/exploration-and-plotting
import pandas as pd

import numpy as np



from shapely.wkt import loads as wkt_loads

from shapely import affinity

from matplotlib.patches import Polygon

import matplotlib.pyplot as plt

import matplotlib.image as mpimg



import cv2



df = pd.read_csv('../input/train_wkt_v3.csv')

df.head(5)
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
gs = pd.read_csv('../input/grid_sizes.csv', names=['ImageId', 'Xmax', 'Ymin'], skiprows=1)

print(gs.head())
df['ImageId'].unique()
# First Image

polygonsList ={}

image = df[df.ImageId == '6100_1_3']

for cType in image.ClassType.unique():

    polygonsList[cType] = wkt_loads(image[image.ClassType == cType].MultipolygonWKT.values[0])
polygonsList
# plot using matplotlib

fig, ax = plt.subplots(figsize=(8, 8))



# plotting, color by class type

for p in polygonsList:

    for polygon in polygonsList[p]:

        mpl_poly = Polygon(np.array(polygon.exterior), color=plt.cm.Set1(p*10), lw=0, alpha=0.3)

        ax.add_patch(mpl_poly)



ax.relim()

ax.autoscale_view()
# Anzahl Features



for p in polygonsList:

    print("Type: {:4d}, objects: {}".format(p,len(polygonsList[p].geoms)))
df['polygons'] = df.apply(lambda row: wkt_loads(row.MultipolygonWKT),axis=1)

df['nPolygons'] = df.apply(lambda row: len(row['polygons'].geoms), axis = 1)



pvt = df.pivot(index='ImageId', columns='ClassType', values='nPolygons')

pvt
from os import listdir

imagenames_16 = listdir('../input/sixteen_band') 

imagenames_13 = listdir('../input/three_band') 
###############################

# Convert polygons to pixels  #

###############################
def _get_image_names(base_path, imageId):

    '''

    Get the names of the tiff files

    '''

    d = {'3': path.join(base_path,'three_band/{}.tif'.format(imageId)),             # (3, 3348, 3403)

         'A': path.join(base_path,'sixteen_band/{}_A.tif'.format(imageId)),         # (8, 134, 137)

         'M': path.join(base_path,'sixteen_band/{}_M.tif'.format(imageId)),         # (8, 837, 851)

         'P': path.join(base_path,'sixteen_band/{}_P.tif'.format(imageId)),         # (3348, 3403)

         }

    return d
def _convert_coordinates_to_raster(coords, img_size, xymax):

    Xmax,Ymax = xymax

    H,W = img_size

    W1 = 1.0*W*W/(W+1)

    H1 = 1.0*H*H/(H+1)

    xf = W1/Xmax

    yf = H1/Ymax

    coords[:,1] *= yf

    coords[:,0] *= xf

    coords_int = np.round(coords).astype(np.int32)

    return coords_int

def _get_xmax_ymin(grid_sizes_panda, imageId):

    xmax, ymin = grid_sizes_panda[grid_sizes_panda.ImageId == imageId].iloc[0,1:].astype(float)

    return (xmax,ymin)
def _get_polygon_list(wkt_list_pandas, imageId, cType):

    df_image = wkt_list_pandas[wkt_list_pandas.ImageId == imageId]

    multipoly_def = df_image[df_image.ClassType == cType].MultipolygonWKT

    polygonList = None

    if len(multipoly_def) > 0:

        assert len(multipoly_def) == 1

        polygonList = wkt_loads(multipoly_def.values[0])

    return polygonList
def _get_and_convert_contours(polygonList, raster_img_size, xymax):

    perim_list = []

    interior_list = []

    if polygonList is None:

        return None

    for k in range(len(polygonList)):

        poly = polygonList[k]

        perim = np.array(list(poly.exterior.coords))

        perim_c = _convert_coordinates_to_raster(perim, raster_img_size, xymax)

        perim_list.append(perim_c)

        for pi in poly.interiors:

            interior = np.array(list(pi.coords))

            interior_c = _convert_coordinates_to_raster(interior, raster_img_size, xymax)

            interior_list.append(interior_c)

    return perim_list,interior_list
def _plot_mask_from_contours(raster_img_size, contours, class_value = 1):

    img_mask = np.zeros(raster_img_size,np.uint8)

    if contours is None:

        return img_mask

    perim_list,interior_list = contours

    cv2.fillPoly(img_mask,perim_list,class_value)

    cv2.fillPoly(img_mask,interior_list,0)

    return img_mask
def generate_mask_for_image_and_class(raster_size, imageId, class_type, grid_sizes_panda,

                                     wkt_list_pandas):

    xymax = _get_xmax_ymin(grid_sizes_panda,imageId)

    polygon_list = _get_polygon_list(wkt_list_pandas,imageId,class_type)

    contours = _get_and_convert_contours(polygon_list,raster_size,xymax)

    mask = _plot_mask_from_contours(raster_size,contours,1)

    return mask
set_of_mask = dict()

mask_test = np.zeros((500,500))



for i in range(0,9):

    

    mask = generate_mask_for_image_and_class((500,500),"6100_1_3",i,gs,df)

    set_of_mask[i] =  mask*255/9*i

    mask_test = mask_test + mask*255/9*i

    

cv2.imwrite("mask.png",mask_test)

img = mpimg.imread('mask.png')

plt.imshow(img)

import tifffile as tiff



img_filename = '../input/three_band/6100_1_3.tif'



Image = tiff.imread(img_filename)



tiff.imshow(Image)

np.shape(Image)
#####################################################################

# The 3 next In's show how to scale polygons to image size and back #

#####################################################################

img_id = "6100_1_3"

i_grid_size = gs[gs.ImageId == img_id]

x_max = i_grid_size.Xmax.values[0]

y_min = i_grid_size.Ymin.values[0]



# Get just single class of trianing polyongs for this image

class_2 = df[(df.ImageId == img_id) & (df.ClassType == 2)]



# WKT to shapely object

polyg = wkt_loads(class_2.MultipolygonWKT.values[0])



print('Original Extent')

print(polyg.bounds)
#Load the image and get its width and height



#image = gdal.Open('three_band/6120_2_2.tif')

#W = image.RasterXSize

#H = image.RasterYSize

#gdal is not loaded in kaggle yet, so I'll do these manually for now.



W = 3403

H = 3348



# Transform the polygons



W_ = W * (W / (W+1) )

H_ = H * (H / (H+1) )



x_scaler = W_ / x_max

y_scaler = H_ / y_min



polyg = affinity.scale(polyg, xfact = x_scaler, yfact = y_scaler, origin=(0,0,0))



print("New Extent to match raster")

print(polyg.bounds)
# Now scale the shapely file back to its original coordinates for submission 

# The scaler is the inverse of the original scaler

x_scaler = 1 / x_scaler

y_scaler = 1 / y_scaler



polyg = affinity.scale(polyg, xfact = x_scaler, yfact = y_scaler, origin=(0,0,0))

print("Back to original")

print(polyg.bounds)
############################

# SOBEL EDGE DETECTION #

# from https://www.kaggle.com/bkamphaus/draper-satellite-image-chronology/exploratory-image-analysis #

############################





import skimage

from skimage.feature import greycomatrix, greycoprops

from skimage.filters import sobel, sobel_h, sobel_v



# load the image and convert it to grayscale

image = Image





dims = np.shape(image)

print(dims)



# a sobel filter is a basic way to get an edge magnitude/gradient image

tiff.imshow(image)

tiff.imshow(sobel(image[2,:750,:750]))

#tiff.imshow(sobel_h(image[2,:750,:750]), cmap='BuGn')    

#tiff.imshow(sobel_v(image[2,:750,:750]), cmap='BuGn')    

from sklearn.decomposition import PCA



pca = PCA(3)

pca.fit(image.matrix)

image_pca = pca.transform(image.matrix)

image_pca_img = image.to_matched_img(image_pca)



tiff.imshow(image_pca_img)
#############################

#

#############################



from skimage import color



hsv = color.rgb2hsv(image)