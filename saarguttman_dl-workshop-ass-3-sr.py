import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import json
import imageio
import math
import seaborn as sns
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Flatten, Concatenate, Input, Conv2D, AvgPool2D, MaxPool2D, Dropout, MaxPooling2D, BatchNormalization, UpSampling2D, Add, Activation, LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.applications import vgg16
from keras.optimizers import Adam, SGD
from keras import backend as K
from keras.regularizers import l2
from keras import regularizers
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score,log_loss
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from scipy import misc
from keras.applications.vgg16 import VGG16
from keras.layers import Lambda
import tensorflow as tf
from skimage.transform import resize
%matplotlib inline
# Define our custom metric
def PSNR(y_true, y_pred):
#     max_pixel = 255.0
#     return 10.0 * math.log10((max_pixel ** 2) / (K.mean(K.square(y_pred - y_true)))) 
    max_pixel = 1.0
    return 10.0 * (1.0 / math.log(10)) * K.log((max_pixel ** 2) / (K.mean(K.square(y_pred/255.0 - y_true/255.0))))

# Define our custom loss function
def inverse_PSNR(y_true, y_pred):
    return 1.0/PSNR(y_true, y_pred)


# Define our custom loss function
def inverse_PSNR_x10(y_true, y_pred):
    return 100.0/PSNR(y_true, y_pred) 


# Define our custom loss function
def charbonnier(y_true, y_pred):
    epsilon = 1e-3
    error = y_true - y_pred
    p = K.sqrt(K.square(error) + K.square(epsilon))
    return K.mean(p)

def plot_result(plot_size, pics, titles):
    f, axarr = plt.subplots(1, len(pics), figsize=plot_size)
    
    for i in range(len(pics)):
        axarr[i].imshow(pics[i])
        axarr[i].title.set_text(titles[i])
        
def load_img(name):
    img = imageio.imread(name)
    return img

def load_images(images):
    img_directory = '../input/pascal-voc-2007/voctrainval_06-nov-2007/VOCdevkit/VOC2007/JPEGImages/'
    arr = os.listdir(img_directory)
    for i in range(1000):
        img = load_img(img_directory+arr[i])
        images.append(img)

images = []
load_images(images)
x_small = []
size = 72
for img in images:
    resized = resize(img, (size, size))
    x_small.append(resized)

x_medium = []
size = 144
for img in images:
    resized = resize(img, (size, size))
    x_medium.append(resized)
    
x_large = []
size = 288
for img in images:
    resized = resize(img, (size, size))
    x_large.append(resized)
    
x_small = np.asarray(x_small)
x_medium = np.asarray(x_medium)
x_large = np.asarray(x_large)
x_train = np.asarray(x_small[:800])
y_medium_train = np.asarray(x_medium[:800])
y_large_train = x_large[:800]

x_validation = x_small[800:]
y_medium_validation = x_medium[800:]
y_large_validation = x_large[800:]
pics = [x_train[0], y_medium_train[0], y_large_train[0]]
titles = ['72X72', '144X144', '288X288']
plot_result((20,9), pics, titles)

def model_1(loss_func=inverse_PSNR_x10):
    inp = Input((72,72,3))
    x = Conv2D(64,(3,3),activation='relu',padding='SAME')(inp)
    x = Conv2D(64,(3,3),activation='relu', padding='SAME')(x)
    x = UpSampling2D(2)(x)
    x = Conv2D(3, (1,1),activation='relu', padding='SAME')(x)

    model = Model(inputs=inp,outputs=x)
    model.summary()
    model.compile(loss=[inverse_PSNR_x10], metrics=[PSNR], optimizer='adam')
    
    return model
model = model_1()
model.fit(x_train, y_medium_train, validation_data=[x_validation, y_medium_validation], epochs=50)
res_PSNR = model.predict(np.asarray([x_validation[0]]))

pics_PSNR = [x_validation[0], res_PSNR[0], y_medium_validation[0]]
titles_PSNR = ['72X72 ipnut', '144X144 Prediction', '144X144 Actual']
plot_result((20,9), pics_PSNR, titles_PSNR)

model = model_1('rmse')
model.fit(x_train, y_medium_train, validation_data=[x_validation, y_medium_validation], epochs=50)
res = model.predict(np.asarray([x_validation[0]]))
pics = [x_validation[0], res_PSNR[0], res[0], y_medium_validation[0]]
titles = ['72X72 ipnut', '144X144 PSNR Prediction', '144X144 RMSE Prediction', '144X144 Actual']
plot_result((30,9), pics, titles)
inp = Input((None,None,3))
x = Conv2D(64,(3,3),activation='relu',padding='SAME')(inp)
x = Conv2D(64,(3,3),activation='relu', padding='SAME')(x)
x = UpSampling2D(2)(x)
med = Conv2D(3, (1,1),activation='relu', padding='SAME')(x)

x2 = UpSampling2D(2)(x)
x2 = Conv2D(3, (1,1),activation='relu', padding='SAME')(x2)

model = Model(inputs=inp,outputs=[med, x2])
model.summary()
model.compile(loss="mse", metrics=[PSNR], optimizer='adam')
model.fit(x_train, [y_medium_train, y_large_train], validation_data=[x_validation, [y_medium_validation, y_large_validation]], epochs=50)
res = model.predict(np.asarray([x_validation[0]]))

pics = [x_validation[0], res[0][0], y_medium_validation[0], res[1][0], y_large_validation[0]]
titles = ['72X72 ipnut', '144X144 Prediction', '144X144 Actual', '288X288 Prediction', '288X288 Actual']
plot_result((20,9), pics, titles)
def res_block(size=None):
    inp = Input((size,size,32))
    x = Conv2D(32,(3,3),activation='relu',padding='SAME')(inp)
    x = Conv2D(32,(3,3),activation='relu', padding='SAME')(x)
    x = Add()([x, inp])
    x = Activation('relu')(x)

    model = Model(inputs=inp,outputs=x)
    return model

inp = Input((None,None,3))
x = Conv2D(32,(1,1),activation='relu',padding='SAME')(inp)
x = res_block()(x)
x = res_block()(x)
x = res_block()(x)
x = res_block()(x)
x = res_block()(x)
x = res_block()(x)
x = res_block()(x)
x = res_block()(x)
x = UpSampling2D(2)(x)
med = Conv2D(3, (1,1),activation='relu', padding='SAME')(x)

x2 = UpSampling2D(2)(x)
x2 = Conv2D(3, (1,1),activation='relu', padding='SAME')(x2)

model = Model(inputs=inp,outputs=[med, x2])
model.summary()
model.compile(loss=inverse_PSNR_x10, metrics=[PSNR], optimizer='adam')
model.fit(x_train, [y_medium_train, y_large_train], validation_data=[x_validation, [y_medium_validation, y_large_validation]], batch_size=32,  epochs=100)
res = model.predict(np.asarray([x_validation[0]]))

pics = [x_validation[0], res[0][0], y_medium_validation[0], res[1][0], y_large_validation[0]]
titles = ['72X72 ipnut', '144X144 Prediction', '144X144 Actual', '288X288 Prediction', '288X288 Actual']
plot_result((20,9), pics, titles)
def dilated_block(cnl=32):
    inp = Input((None,None,32))
    x1 = Conv2D(32, (3, 3), activation=LeakyReLU(0.2), padding='SAME', dilation_rate=1)(inp)
    x2 = Conv2D(32, (3, 3), activation=LeakyReLU(0.2), padding='SAME', dilation_rate=2)(inp)
    x3 = Conv2D(32, (3, 3), activation=LeakyReLU(0.2), padding='SAME', dilation_rate=4)(inp)
    x = Concatenate()([x1, x2, x3])
    x = Activation(LeakyReLU(0.2))(x)
    x = Conv2D(32,(3,3),activation=LeakyReLU(0.2),padding='SAME')(x)

    model = Model(inputs=inp,outputs=x)
    return model

inp = Input((None,None,3))
x = Conv2D(32,(1,1),activation=LeakyReLU(0.2),padding='SAME')(inp)
x = dilated_block()(x)
x = dilated_block()(x)
x = UpSampling2D(2)(x)
med = Conv2D(3, (1,1),activation=LeakyReLU(0.2), padding='SAME')(x)

x2 = UpSampling2D(2)(x)
x2 = Conv2D(3, (1,1),activation=LeakyReLU(0.2), padding='SAME')(x2)

model = Model(inputs=inp,outputs=[med, x2])
model.summary()
model.compile(loss=inverse_PSNR_x10, metrics=[PSNR], optimizer='adam')
model.fit(x_train, [y_medium_train, y_large_train], validation_data=[x_validation, [y_medium_validation, y_large_validation]], epochs=50)
res = model.predict(np.asarray([x_validation[0]]))

pics = [x_validation[0], res[0][0], y_medium_validation[0], res[1][0], y_large_validation[0]]
titles = ['72X72 ipnut', '144X144 Prediction', '144X144 Actual', '288X288 Prediction', '288X288 Actual']
plot_result((20,9), pics, titles)

vgg_model = VGG16(weights='imagenet', include_top=False, input_shape = (None,None, 3))
vgg_model.trainable = False
for layer in vgg_model.layers:
    layer.trainable = False
vgg_layer = Model(inputs=vgg_model.input, outputs=vgg_model.get_layer("block1_conv2").output)

    
inp = Input((None,None,3))
x = Conv2D(64,(3,3),activation=LeakyReLU(0.2),padding='SAME')(inp)
x = Conv2D(64,(3,3),activation=LeakyReLU(0.2),padding='SAME')(x)


model2 = vgg_layer(inp) 
x = Concatenate()([x,model2]) 

x = UpSampling2D(2)(x)
med = Conv2D(3, (1,1),activation=LeakyReLU(0.2), padding='SAME')(x)

x2 = UpSampling2D(2)(x)
x2 = Conv2D(3, (1,1),activation=LeakyReLU(0.2), padding='SAME')(x2)

model = Model(inputs=inp,outputs=[med, x2])
model.summary()

model.compile(loss='mse', metrics=[PSNR], optimizer='adam')
model.fit(x_train, [y_medium_train, y_large_train], validation_data=[x_validation, [y_medium_validation, y_large_validation]], epochs=50)
res = model.predict(np.asarray([x_validation[0]]))

pics = [x_validation[0], res[0][0], y_medium_validation[0], res[1][0], y_large_validation[0]]
titles = ['72X72 ipnut', '144X144 Prediction', '144X144 Actual', '288X288 Prediction', '288X288 Actual']
plot_result((20,9), pics, titles)
def pixel_shuffle(scale=2):
    return lambda x: tf.nn.depth_to_space(x, scale)
vgg_model = VGG16(weights='imagenet', include_top=False, input_shape = (None,None, 3))
vgg_model.trainable = False
for layer in vgg_model.layers:
    layer.trainable = False
vgg_layer = Model(inputs=vgg_model.input, outputs=vgg_model.get_layer("block1_conv2").output)


    
inp = Input((None,None,3))
x = Conv2D(64,(3,3),activation=LeakyReLU(0.2),padding='SAME')(inp)
x = Conv2D(64,(3,3),activation=LeakyReLU(0.2),padding='SAME')(x)


model2 = vgg_layer(inp) 
x = Concatenate()([x,model2]) 

x = Lambda(pixel_shuffle())(x)
x = Conv2D(4,(1,1),activation=LeakyReLU(0.2),padding='SAME')(x)
med = Conv2D(3, (1,1),activation=LeakyReLU(0.2), padding='SAME')(x)

x = Conv2D(64,(3,3),activation=LeakyReLU(0.2),padding='SAME')(x)
x = Conv2D(64,(3,3),activation=LeakyReLU(0.2),padding='SAME')(x)

x2 = Lambda(pixel_shuffle())(x)
x = Conv2D(4,(1,1),activation=LeakyReLU(0.2),padding='SAME')(x)
x2 = Conv2D(3, (1,1),activation=LeakyReLU(0.2), padding='SAME')(x2)

model = Model(inputs=inp,outputs=[med, x2])
model.summary()

model.compile(loss=inverse_PSNR_x10, metrics=[PSNR], optimizer='adam', loss_weights=[0.7, 0.3],)
model.fit(x_train, [y_medium_train, y_large_train], validation_data=[x_validation, [y_medium_validation, y_large_validation]], epochs=50)
res = model.predict(np.asarray([x_validation[0]]))

pics = [x_validation[0], res[0][0], y_medium_validation[0], res[1][0], y_large_validation[0]]
titles = ['72X72 ipnut', '144X144 Prediction', '144X144 Actual', '288X288 Prediction', '288X288 Actual']
plot_result((20,9), pics, titles)
def vgg_loss(y_true, y_pred):
    vgg_model = VGG16(weights='imagenet', include_top=False, input_shape = (None,None, 3))
    vgg_model.trainable = False
    for layer in vgg_model.layers:
        layer.trainable = False
    loss_model = Model(inputs=vgg_model.input, outputs=vgg_model.get_layer("block1_conv2").output)
    loss_model.trainable = False
    
    return K.mean(K.square(loss_model(y_true) - loss_model(y_pred)))

vgg_model = VGG16(weights='imagenet', include_top=False, input_shape = (None,None, 3))
vgg_model.trainable = False
for layer in vgg_model.layers:
    layer.trainable = False
vgg_layer = Model(inputs=vgg_model.input, outputs=vgg_model.get_layer("block1_conv2").output)


    
inp = Input((None,None,3))
x = Conv2D(64,(3,3),activation=LeakyReLU(0.2),padding='SAME')(inp)
x = Conv2D(64,(3,3),activation=LeakyReLU(0.2),padding='SAME')(x)


model2 = vgg_layer(inp) 
x = Concatenate()([x,model2]) 

x = Lambda(pixel_shuffle())(x)
x = Conv2D(4,(1,1),activation=LeakyReLU(0.2),padding='SAME')(x)
med = Conv2D(3, (1,1),activation=LeakyReLU(0.2), padding='SAME')(x)

x = Conv2D(64,(3,3),activation=LeakyReLU(0.2),padding='SAME')(x)
x = Conv2D(64,(3,3),activation=LeakyReLU(0.2),padding='SAME')(x)

x2 = Lambda(pixel_shuffle())(x)
x = Conv2D(4,(1,1),activation=LeakyReLU(0.2),padding='SAME')(x)
x2 = Conv2D(3, (1,1),activation=LeakyReLU(0.2), padding='SAME')(x2)

model = Model(inputs=inp,outputs=[med, x2])
model.summary()

model.compile(loss=vgg_loss, metrics=[PSNR], optimizer='adam', loss_weights=[0.7, 0.3],)
model.fit(x_train, [y_medium_train, y_large_train], validation_data=[x_validation, [y_medium_validation, y_large_validation]], epochs=50)
res = model.predict(np.asarray([x_validation[0]]))

pics = [x_validation[0], res[0][0], y_medium_validation[0], res[1][0], y_large_validation[0]]
titles = ['72X72 ipnut', '144X144 Prediction', '144X144 Actual', '288X288 Prediction', '288X288 Actual']
plot_result((20,9), pics, titles)