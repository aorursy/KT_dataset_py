import pandas as pd
import numpy as np
from shapely import wkt
from shapely import affinity
import shapely
import gdal

polygons_raw = pd.read_csv('../input/train_wkt_v3.csv')
grid_sizes = pd.read_csv('../input/grid_sizes.csv')
cols = grid_sizes.columns.tolist()
cols[0]='ImageId'
grid_sizes.columns = cols

img_id = '6120_2_2'
i_grid_size = grid_sizes[grid_sizes.ImageId == img_id]
x_max = i_grid_size.Xmax.values[0]
y_min = i_grid_size.Ymin.values[0]

#Get just a single class of training polygons for this image
class_2 = polygons_raw[(polygons_raw.ImageId == img_id) & (polygons_raw.ClassType==2)]

#WKT to shapely object
polygons = wkt.loads(class_2.MultipolygonWKT.values[0])

print('Original Extent')
print(polygons.bounds)

#Load the image and get its width and height
#image = gdal.Open('three_band/6120_2_2.tif')
#W = image.RasterXSize
#H = image.RasterYSize
#gdal is not loaded in kaggle yet, so I'll do these manually for now.
W = 3403
H = 3348

#Transform the polygons 
W_ = W * (W/(W+1))
H_ = H * (H/(H+1))

x_scaler = W_ / x_max
y_scaler = H_ / y_min

polygons = shapely.affinity.scale(polygons, xfact = x_scaler, yfact= y_scaler, origin=(0,0,0))

print('New Extent to match raster')
print(polygons.bounds)

#Now scale the shapely file back to its original coordinates for submission
#The scaler is the inverse of the original scaler
x_scaler = 1/x_scaler
y_scaler = 1/y_scaler

polygons = shapely.affinity.scale(polygons, xfact = x_scaler, yfact= y_scaler, origin=(0,0,0))

print('Back to original')
print(polygons.bounds)