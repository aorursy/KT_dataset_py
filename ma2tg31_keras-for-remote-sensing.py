from collections import defaultdict

import csv

import sys



import cv2

from shapely.geometry import MultiPolygon, Polygon

import shapely.wkt

import shapely.affinity

import numpy as np

import tifffile as tiff



csv.field_size_limit(sys.maxsize);
IM_ID = '6120_2_2'

POLY_TYPE = '1'  # buildings



# Load grid size

x_max = y_min = None

for _im_id, _x, _y in csv.reader(open('../input/grid_sizes.csv')):

    if _im_id == IM_ID:

        x_max, y_min = float(_x), float(_y)

        break



# Load train poly with shapely

train_polygons = dict()

for _im_id, _poly_type, _poly in csv.reader(open('../input/train_wkt_v4.csv')):

    if _im_id == IM_ID:

        train_polygons[_poly_type]=shapely.wkt.loads(_poly)

        break



# Read image with tiff

im_rgb = tiff.imread('../input/three_band/{}.tif'.format(IM_ID)).transpose([1, 2, 0])

im_size = im_rgb.shape[:2]



def get_scalers():

    h, w = im_size  # they are flipped so that mask_for_polygons works correctly

    w_ = w * (w / (w + 1))

    h_ = h * (h / (h + 1))

    return w_ / x_max, h_ / y_min



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

mask_map = list()

for key in train_polygons.keys():

    x_scaler, y_scaler = get_scalers()

    train_polygons_scaled = shapely.affinity.scale(train_polygons[key], xfact=x_scaler, yfact=y_scaler, origin=(0, 0, 0))

    mask_map.append( mask_for_polygons(train_polygons_scaled) )

    

mask = np.dstack(mask_map)
nrow, ncol = im_size

x = np.arange(0, nrow, 16)

y = np.arange(0, ncol, 16)



patch_list = list()

lbl_list = list()

for xstart, xend in zip(x[:-1], x[1:]):

    for ystart, yend in zip(y[:-1], y[1:]):

        patch_list.append(im_rgb[xstart:xend, ystart:yend,:])

        lbl_list.append(mask[xstart:xend, ystart:yend].mean())

patches = np.array(patch_list)

lbls = np.array(lbl_list)
from sklearn.cross_validation import train_test_split



xtrain, xtest, ytrain, ytest = train_test_split(patches, lbls,

                                  train_size=0.8,

                                  test_size=0.2)
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense

from keras.layers.normalization import BatchNormalization

from keras.layers import Activation

from keras.models import Sequential



model = Sequential()

model.add(Convolution2D(64, 1, 1, border_mode='same', input_shape=(16, 16, 3), dim_ordering="tf"))

model.add(BatchNormalization(mode=0))

model.add(Activation("relu"))

model.add(MaxPooling2D((2,2), dim_ordering="tf"))

model.add(Convolution2D(64, 1, 1, border_mode='same', dim_ordering="tf"))

model.add(BatchNormalization(mode=0))

model.add(Activation("relu"))

model.add(Convolution2D(128, 3, 3, border_mode='same', dim_ordering="tf"))

model.add(BatchNormalization(mode=0))

model.add(Activation("relu"))

model.add(MaxPooling2D((2,2), dim_ordering="tf"))

model.add(Flatten())

model.add(Dense(1))



model.compile(loss='mean_squared_error', optimizer='adam')
%time model.fit(xtrain.values, ytrain.values, batch_size=32, nb_epoch=10, verbose=False)



model.evaluate(xtrain.values, ytrain.values, batch_size=32, verbose=False), model.evaluate(X_test.values, Y_test.values, batch_size=32, verbose=False)