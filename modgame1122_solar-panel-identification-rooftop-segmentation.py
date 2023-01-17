# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
!pip install rasterio
import os
import cv2
import json
import time
import random
import shapely.ops
import numpy as np
import pandas as pd
import math as Math
import tensorflow as tf
import shapely.geometry
import rasterio.features
import keras.backend as K
from skimage import filters
from skimage.io import imread
import matplotlib.pyplot as plt
from skimage.util import montage
from matplotlib.path import Path
from keras.optimizers import Adam
from urllib.request import urlopen
from skimage.color import label2rgb
from skimage.transform import resize
from keras import models, layers, Model
from keras.losses import binary_crossentropy
from matplotlib.collections import PatchCollection
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator

data_dir = os.path.join('AerialImageDataset', 'train')
nldata_dir = os.path.join('AerialImageDataset', 'test')

BATCH_SIZE = 64
MIN_POLYGON_AREA = 150
IMAGE_SIZE = (256, 256)
PANEL_SIZE = (1.651, 0.9906)
PANEL_POWER = 255
def batch_img_gen(directory, dir_list, batch_size, cropping = -1, seg = True):
    kernel = np.ones((3,3),np.uint8)
    #predtestmod[0, :, :, 0] = cv2.erode(predtestmod[0, :, :, 0].astype('uint8'),kernel,iterations = 2)
    out_img, out_seg = [], []
    while True:
        for imgname in np.random.permutation(dir_list):
            img_data = np.asarray(imread(os.path.join(directory, 'images', imgname)))
            if(seg):
                seg_data = np.expand_dims(cv2.erode(imread(os.path.join(directory, 'gt', imgname)),kernel,iterations = 1), -1)
            if (cropping == -1):
                crop = random.randint(200,400)
            else:
                crop = cropping
            for i in range(int((5000/crop)) - 1):
                for j in range(int((5000/crop)) - 1):
                    out_img += [resize(img_data[i * crop:(i + 1) * crop, j * crop:(j + 1) * crop], (IMAGE_SIZE[0], IMAGE_SIZE[1]))]
                    if(seg):
                        out_seg += [resize(seg_data[i * crop:(i + 1) * crop, j * crop:(j + 1) * crop], (IMAGE_SIZE[0], IMAGE_SIZE[1]))]
                    if len(out_img)>=batch_size:
                        yield (np.stack(out_img, 0)).astype(np.float32), np.stack(out_seg, 0).astype(np.float32) if seg else None
                        out_img.clear()
                        out_seg.clear()
                        
def hist_equi(img_data):
    img2 = cv2.cvtColor(img_data, cv2.COLOR_RGB2YCR_CB)
    equ = cv2.equalizeHist(img2[:,:,0])
    img2[:,:,0] = equ
    img2 = cv2.cvtColor(img2, cv2.COLOR_YCR_CB2RGB)
    return img2
                        
def mask_to_poly(mask, min_polygon_area_th=MIN_POLYGON_AREA):
    mask = (mask > 0.5).astype(np.uint8)
    shapes = rasterio.features.shapes(mask.astype(np.int16), mask > 0)
    poly_list = []
    for shape, value in shapes:
        if(shapely.geometry.shape(shape).is_valid == False):
            poly_list += [shapely.geometry.shape(shape).buffer(0)]
        else:
            poly_list += [shapely.geometry.shape(shape)]
    mp = shapely.ops.cascaded_union(
        shapely.geometry.MultiPolygon(poly_list))
    if isinstance(mp, shapely.geometry.Polygon):
        if(mp.area > min_polygon_area_th):
            return mp
    else:
        return shapely.geometry.MultiPolygon([P for P in mp if P.area > min_polygon_area_th])

                        
def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    return K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)

def dice_p_bce(in_gt, in_pred):
    return 0.05*binary_crossentropy(in_gt, in_pred) - dice_coef(in_gt, in_pred)

def true_positive_rate(y_true, y_pred):
    return K.sum(K.flatten(y_true)*K.flatten(K.round(y_pred)))/K.sum(y_true)
model = models.load_model('/kaggle/input/FinalModelSeg.h5', custom_objects = {'dice_p_bce' : dice_p_bce, 'dice_coef' : dice_coef, 'true_positive_rate' : true_positive_rate})
location = "Alessandro-Volta-Platz 1, 34123 Kassel, Germany"



loc = ""
for i in location:
    if i == ' ':
        loc += '%20'
    else:
        loc += i
import urllib.request, json 
with urllib.request.urlopen("https://api.mapbox.com/geocoding/v5/mapbox.places/"+loc+".json?access_token=pk.eyJ1IjoibWF0dGZpY2tlIiwiYSI6ImNqNnM2YmFoNzAwcTMzM214NTB1NHdwbnoifQ.Or19S7KmYPHW8YjRz82v6g&cachebuster=1564656756986&autocomplete=true&limit=10") as url:
    data = json.loads(url.read().decode())
for i in range(len(data['features'])):
    print(i + 1,data['features'][i]['place_name'])
loc_num = 1
zoom = 17
shift = (0, 0)

url = "https://api.mapbox.com/styles/v1/mapbox/satellite-v9/static/" + str(data['features'][loc_num - 1]['center'][0] + (shift[0] / 100000)) + "," + str(data['features'][loc_num - 1]['center'][1] + (shift[1] / 100000)) + "," + str(zoom) + ",0,0/280x280?access_token=pk.eyJ1IjoibW9kZ2FtZTExMjIiLCJhIjoiY2p5c2VvMWhzMGwzbzNtbzZ0ZGE3N3JkNiJ9.b_Ks63vMGOkPAA9SedTq0Q"
#url = "http://maps.googleapis.com/maps/api/staticmap?center=" + str(data['features'][loc_num - 1]['center'][0] + (shift[0] / 100000)) + "," + str(data['features'][loc_num - 1]['center'][1] + (shift[1] / 100000)) + "&size=800x800&zoom=14&sensor=false&key=AIzaSyAv4HykTKbQz4DgAArAr2CcgsEuQCejFns"
img_data = imread(url)
start = time.time()
meters_img_ratio = 78730.74269493636 * Math.cos(data['features'][loc_num - 1]['center'][1] * 3.14159 / 180) / Math.pow(2, zoom) * IMAGE_SIZE[0]
montage_rgb = lambda x: np.stack([montage(x[:, :, :, i]) for i in range(x.shape[3])], -1)
center = shapely.geometry.Point(IMAGE_SIZE[0] / 2,IMAGE_SIZE[1] / 2)
circle = center.buffer(75 * 50 / meters_img_ratio)
geo_size = (meters_img_ratio, meters_img_ratio)
p_size_img = ((PANEL_SIZE[0] / geo_size[0]) * IMAGE_SIZE[0], (PANEL_SIZE[1] / geo_size[1]) * IMAGE_SIZE[1])
poly_contain = None
poly_intersect = None
area = 0
count = 0
max_int_area = 0
fig, (ax1, ax2, ax3, c_ax, d_ax) = plt.subplots(1, 5, figsize = (40,60))
test = [resize(img_data[:, :, 0:3], (256, 256))]
# test = [resize(img_data[0:int(img_data.shape[0] / 2), 0:int(img_data.shape[1] / 2)], (256, 256)),
#         resize(img_data[0:int(img_data.shape[0] / 2), int(img_data.shape[1] / 2):], (256, 256)),
#         resize(img_data[int(img_data.shape[0] / 2):, 0:int(img_data.shape[1] / 2)], (256, 256)),
#         resize(img_data[int(img_data.shape[0] / 2):, int(img_data.shape[1] / 2):], (256, 256))]
test = np.asarray(test)
predtest = model.predict(np.asarray(test))
predtestmod = np.where((predtest * 255) > 254.1, 255, 0)
#predtestmod[0, :, :, 0] = filters.threshold_minimum(predtestmod[0, :, :, 0])
#predtestmod[0, :, :, 0] = cv2.adaptiveThreshold(predtestmod[0, :, :, 0].astype('uint8') ,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
#predtestmod = np.where((predtestmod) > 254.9, 0, 255)
polygons = mask_to_poly(montage(predtestmod[:, :, :, 0]))
ax1.imshow(montage_rgb(test))
ax2.imshow(montage(predtest[:, :, :, 0]),  cmap = 'bone_r')
ax3.imshow(montage(predtestmod[:, :, :, 0]), cmap = 'bone_r')
c_ax.imshow(montage(predtestmod[:, :, :, 0]*0), cmap = 'bone_r')
d_ax.imshow(montage(predtestmod[:, :, :, 0]*0), cmap = 'bone_r')
if isinstance(polygons, shapely.geometry.Polygon):
    if(polygons.contains(center)):
        poly_contain = polygons
    if (polygons.intersection(circle).area > max_int_area):
        poly_intersect = polygons
    x, y = polygons.exterior.coords.xy
    c_ax.plot(x, y)
elif len(polygons) > 0:
    for poly in polygons:
        if(poly.is_valid == False):
            poly = poly.buffer(0)
        if(poly.contains(center)):
            poly_contain = poly
        if(poly_contain == None) and (poly.intersection(circle).area > max_int_area):
            max_int_area = poly.intersection(circle).area
            poly_intersect = poly
        x, y = poly.exterior.coords.xy
        c_ax.plot(x, y)
        
area = 0
if(poly_contain != None):
    area = poly_contain.area
    area_covered = count * PANEL_SIZE[0] * PANEL_SIZE[1]
    x, y = poly_contain.exterior.coords.xy
    d_ax.plot(x, y)
    (xmin, ymin, xmax, ymax) = poly_contain.bounds
    str_ymin = ymin
    while(xmin < xmax):
        ymin = str_ymin
        while(ymin < ymax):
            p = shapely.geometry.Polygon([[xmin, ymin],[xmin, ymin + p_size_img[1]],[xmin + p_size_img[0], ymin + p_size_img[1]], [xmin + p_size_img[0], ymin]])
            if poly_contain.contains(p):
                count += 1
                x, y = p.exterior.coords.xy
                d_ax.fill(x, y)
            ymin += p_size_img[1]
        xmin += p_size_img[0]
elif(poly_intersect != None):
    area = poly_intersect.area
    area_covered = count * PANEL_SIZE[0] * PANEL_SIZE[1]
    x, y = poly_intersect.exterior.coords.xy
    d_ax.plot(x, y)
    (xmin, ymin, xmax, ymax) = poly_intersect.bounds
    str_ymin = ymin
    while(xmin < xmax):
        ymin = str_ymin
        while(ymin < ymax):
            p = shapely.geometry.Polygon([[xmin, ymin],[xmin, ymin + p_size_img[1]],[xmin + p_size_img[0], ymin + p_size_img[1]], [xmin + p_size_img[0], ymin]])
            if poly_intersect.contains(p):
                count += 1
                x, y = p.exterior.coords.xy
                d_ax.fill(x, y)
            ymin += p_size_img[1]
        xmin += p_size_img[0]
end = time.time()
fig.savefig('prediction')
print("Estimated time for processing", end - start)
print("")
print("Area of house (Ratio to image) = ", area * 100 / (256 * 256), "%")
print("Actual Area of house =", area * geo_size[0] * geo_size[1] / (IMAGE_SIZE[0] * IMAGE_SIZE[1]), "meter sq.")
print("System Area =", count * PANEL_SIZE[0] * PANEL_SIZE[1], "meter sq.")
print("System Capacity = ", PANEL_POWER * count / 1000, "kW")
print("Estimated number of panels = ", count)
print("Estimated annual generation =", PANEL_POWER * count * 8760 / 1000000, "MWh")