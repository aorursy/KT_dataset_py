import os
import cv2
import json
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
img_dir = os.listdir(os.path.join(data_dir, 'images'))
nlimg_dir = os.listdir(os.path.join(nldata_dir, 'images'))
train_dir, valid_dir = train_test_split(img_dir, test_size = 0.05)
train_dir, test_dir = train_test_split(train_dir, test_size = 0.05)
print(len(train_dir), "Train")
print(len(test_dir), "Test")
print(len(valid_dir), "Valid")
valid_gen = batch_img_gen(data_dir, valid_dir, 1)
t_x, t_y = next(valid_gen)
print('x', t_x.shape, t_x.dtype, t_x.min(), t_x.max())
print('y', t_y.shape, t_y.dtype, t_y.min(), t_y.max())
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (20, 10))
montage_rgb = lambda x: np.stack([montage(x[:, :, :, i]) for i in range(x.shape[3])], -1)
ax1.imshow(montage_rgb(t_x))
ax2.imshow(montage(t_y[:, :, :, 0]), cmap = 'bone_r')
inputlayer = layers.Input((256, 256, 3), name = 'ImageRGB')
c1 = layers.Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(inputlayer)
c1 = layers.Dropout(0.1)(c1)
c1 = layers.Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c1)
p1 = layers.MaxPooling2D((2, 2))(c1)
p1 = layers.BatchNormalization()(p1)
 
c2 = layers.Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p1)
c2 = layers.Dropout(0.1)(c2)
c2 = layers.Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c2)
p2 = layers.MaxPooling2D((2, 2))(c2)
p2 = layers.BatchNormalization()(p2)

c3 = layers.Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p2)
c3 = layers.Dropout(0.2)(c3)
c3 = layers.Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c3)
p3 = layers.MaxPooling2D((2, 2))(c3)
p3 = layers.BatchNormalization()(p3)
 
c4 = layers.Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p3)
c4 = layers.Dropout(0.2)(c4)
c4 = layers.Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c4)
p4 = layers.MaxPooling2D(pool_size=(2, 2))(c4)
p4 = layers.BatchNormalization()(p4)
 
c5 = layers.Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p4)
c5 = layers.Dropout(0.3)(c5)
c5 = layers.Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c5)
 
u6 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
u6 = layers.concatenate([u6, c4])
c6 = layers.Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u6)
c6 = layers.Dropout(0.2)(c6)
c6 = layers.Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c6)
c6 = layers.BatchNormalization()(c6)
 
u7 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
u7 = layers.concatenate([u7, c3])
c7 = layers.Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u7)
c7 = layers.Dropout(0.2)(c7)
c7 = layers.Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c7)
c7 = layers.BatchNormalization()(c7)
 
u8 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
u8 = layers.concatenate([u8, c2])
c8 = layers.Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u8)
c8 = layers.Dropout(0.1)(c8)
c8 = layers.Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c8)
c8 = layers.BatchNormalization()(c8)
 
u9 = layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
u9 = layers.concatenate([u9, c1], axis=3)
c9 = layers.Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u9)
c9 = layers.Dropout(0.1)(c9)
c9 = layers.Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c9)
 
outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)
 
model = Model(inputs=[inputlayer], outputs=[outputs])
model.compile(optimizer=Adam(1e-4, decay=1e-6), loss=dice_p_bce, metrics=[dice_coef, 'binary_accuracy', true_positive_rate])
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
weight_path="{}_weights.best5.hdf5".format('seg_model')

checkpoint = ModelCheckpoint(weight_path, monitor='val_dice_coef', verbose=1, 
                             save_best_only=True, mode='max', save_weights_only = True)

reduceLROnPlat = ReduceLROnPlateau(monitor='val_dice_coef', factor=0.5, 
                                   patience=3, 
                                   verbose=1, mode='max', epsilon=0.0001, cooldown=2, min_lr=1e-6)
early = EarlyStopping(monitor="val_dice_coef", 
                      mode="max", 
                      patience=15) # probably needs to be more patient, but kaggle time is limited
callbacks_list = [checkpoint, early, reduceLROnPlat]
valid_gen = batch_img_gen(data_dir, valid_dir, BATCH_SIZE)
model.fit_generator(batch_img_gen(data_dir, train_dir, BATCH_SIZE), steps_per_epoch=1000, epochs=100, validation_data = valid_gen, validation_steps = 100, callbacks=callbacks_list)
model.save('best_model2_temp5.h5')
from keras.models import load_model
model.load_weights('seg_model_weights.best3.hdf5')
test_gen = batch_img_gen(data_dir, test_dir, 1, cropping = 256)
test_x, test_y = next(test_gen)
    
print('x', test_x.shape, test_x.dtype, test_x.min(), test_x.max())
print('y', test_y.shape, test_y.dtype, test_y.min(), test_y.max())
pred_y = model.predict(test_x)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (24, 8))
montage_rgb = lambda x: np.stack([montage(x[:, :, :, i]) for i in range(x.shape[3])], -1)
ax1.imshow(montage_rgb(test_x))
ax2.imshow(montage(test_y[:, :, :, 0]), cmap = 'bone_r')
ax2.set_title('Ground Truth')
ax3.imshow(montage(pred_y[:, :, :, 0]), cmap = 'bone_r')
ax3.set_title('Prediction')
a = tf.Variable(test_y)
b = tf.Variable(pred_y)

auc = tf.metrics.auc(a, b)

sess = tf.Session()
sess.run(tf.initialize_all_variables())
sess.run(tf.initialize_local_variables()) # try commenting this line and you'll get the error
train_auc = sess.run(auc)

print(train_auc[1])
non_seg_data = batch_img_gen(nldata_dir, nlimg_dir, 1, seg = False, cropping = 200)
nltest_x, nltest_y = next(non_seg_data)
center = shapely.geometry.Point(230, 120)
nlpred_y = model.predict(nltest_x)
nlpred_mod = np.where(nlpred_y * 255 > 254, 255, 0)
nlpolygons = mask_to_poly(montage(nlpred_mod[:, :, :, 0]))
fig, (ax1, ax2, ax3, c_ax) = plt.subplots(1, 4, figsize = (24, 8))
montage_rgb = lambda x: np.stack([montage(x[:, :, :, i]) for i in range(x.shape[3])], -1)
ax1.imshow(montage_rgb(nltest_x))
ax2.imshow(montage(nlpred_y[:, :, :, 0]), cmap = 'bone_r')
ax2.set_title('Prediction')
ax3.imshow(montage(nlpred_mod[:, :, :, 0]), cmap = 'bone_r')
ax3.set_title('Mod Prediction')
c_ax.imshow(montage(nlpred_y[:, :, :, 0]*0), cmap = 'bone_r')
c_ax.set_title('Boundary')
poly_contain = None
if isinstance(nlpolygons, shapely.geometry.Polygon) and len(nlpolygons) > 0:
    if(nlpolygons.contains(center)):
        poly_contain = nlpolygons
    x, y = nlpolygons.exterior.coords.xy
    c_ax.plot(x, y)
elif len(nlpolygons) > 0:
    for poly in nlpolygons:
        if(poly.contains(center)):
            poly_contain = poly
        x, y = poly.exterior.coords.xy
        c_ax.plot(x, y)
area = 0
if(poly_contain != None):
    area = poly_contain.area
    
print("Area = ", area)
print("Area Ratio to image = ", area * 100 / (256 * 256), "%")
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
img_data = imread(url)
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
fig.savefig('prediction')
print("Area of house (Ratio to image) = ", area * 100 / (256 * 256), "%")
print("Actual Area of house =", area * geo_size[0] * geo_size[1] / (IMAGE_SIZE[0] * IMAGE_SIZE[1]), "meter sq.")
print("System Area =", count * PANEL_SIZE[0] * PANEL_SIZE[1], "meter sq.")
print("System Capacity = ", PANEL_POWER * count / 1000, "kW")
print("Estimated number of panels = ", count)
print("Estimated annual generation =", PANEL_POWER * count * 8760 / 1000000, "MWh")
import cv2
import numpy as np
fig, (plt1, plt2) = plt.subplots(1, 2, figsize = (20, 10))
img = imread('Capture6.JPG')
img2 = cv2.cvtColor(img, cv2.COLOR_RGB2YCR_CB)
equ = cv2.equalizeHist(img2[:,:,0]) #stacking images side-by-side
img2[:,:,0] = equ
img3 = cv2.cvtColor(img2, cv2.COLOR_YCR_CB2RGB)
plt1.hist(img3[:,:,0].ravel(),256,[0,256], color='r')
plt1.hist(img3[:,:,1].ravel(),256,[0,256], color='g')
plt1.hist(img3[:,:,2].ravel(),256,[0,256], color='b')
img2 = montage_rgb(test_x*255)
plt2.hist(img2[:,:,0].ravel(),256,[0,256], color='r')
plt2.hist(img2[:,:,1].ravel(),256,[0,256], color='g')
plt2.hist(img2[:,:,2].ravel(),256,[0,256], color='b')
h = 0