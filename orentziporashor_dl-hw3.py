import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import math
import json
import imageio
from skimage.transform import resize
import matplotlib.pyplot as plt
from keras.models import Model, Sequential
from keras.layers import Dense, Conv2D, Flatten, Input,MaxPool2D, UpSampling2D, Concatenate
from keras import backend as K
from sklearn.metrics import confusion_matrix,accuracy_score,log_loss
from keras.layers import Add,Activation, LeakyReLU
from keras.optimizers import Adam, SGD
## plot images function
def plot_img(images, titles):
    f, axarr = plt.subplots(1, len(images), figsize=(20,9))
    for i in range(len(images)):
        axarr[i].imshow(images[i])
        axarr[i].set_title(titles[i])
       

## custom metric
def PSNR(y_true, y_pred):
    max_pixel = 1.0
    return 10.0 * (1.0 / math.log(10)) * K.log((max_pixel ** 2) / (K.mean(K.square(y_pred/255.0 - y_true/255.0))))

## get 1000 images
def load_img(name):
    img = imageio.imread(name)
    return img
images = []

def load_images():
    img_directory = '../input/pascal-voc-2007/voctrainval_06-nov-2007/VOCdevkit/VOC2007/JPEGImages/'
    arr = os.listdir(img_directory)
    for i in range(1000):
        img = load_img(img_directory+arr[i])
        images.append(img)

load_images()

x_72 = []
x_144 = []
x_288 = []

for img in images:
    resized_72 = resize(img, (72, 72))    
    resized_144 = resize(img, (144, 144))
    resized_288 = resize(img, (288, 288))
    x_72.append(resized_72)
    x_144.append(resized_144)
    x_288.append(resized_288)

x_72 = np.asarray(x_72)
x_144 = np.asarray(x_144)
x_288 = np.asarray(x_288)



print(x_72.shape)
print(x_144.shape)
print(x_288.shape)
x_72_train = x_72[:800]
y_144_train = x_144[:800]
y_288_train = x_288[:800]

x_72_validation = x_72[800:]
y_144_validation = x_144[800:]
y_288_validation = x_288[800:]
plot_img([x_72_train[0], y_144_train[0], y_288_train[0]], ["72","144","288"])
def create_model_1():
    inp = Input((72,72,3))
    x = Conv2D(64,(3,3),activation='relu', padding='same')(inp)
    x = Conv2D(64,(3,3),activation='relu', padding='same')(x)
    x = UpSampling2D(2)(x)
    x = Conv2D(3, (1,1),activation='relu', padding='same')(x)
    model = Model(inputs=inp,outputs=x)
    model.summary()
    return model
model_1 = create_model_1()
model_1.compile(loss='mean_squared_error', optimizer='adam',metrics=[PSNR])

history = model_1.fit(x_72_train, y_144_train,validation_data=[x_72_validation, y_144_validation], epochs=50)
    
preds = model_1.predict(x_72_validation)
plot_img([x_72_validation[0], y_144_validation[0], preds[0]], ["72","144","prediction"])
def create_model_2():
    inp = Input((None,None,3))
    x = Conv2D(64,(3,3),activation='relu', padding='same')(inp)
    x = Conv2D(64,(3,3),activation='relu', padding='same')(x)
    x = UpSampling2D(2)(x)
    x_144 = Conv2D(3, (1,1),activation='relu', padding='same')(x)
    x_288 = UpSampling2D(2)(x)
    x_288 = Conv2D(3, (1,1),activation='relu', padding='same')(x_288)
    model = Model(inputs=inp,outputs=[x_144,x_288])
    model.summary()
    return model
model_2 = create_model_2()
model_2.compile(loss='mean_squared_error', optimizer='adam',metrics=[PSNR])

history = model_2.fit(x_72_train, [y_144_train, y_288_train], validation_data=[x_72_validation, [y_144_validation,y_288_validation]], epochs=50)    
preds = model_2.predict(x_72_validation)
plot_img([x_72_validation[0], y_144_validation[0], y_288_validation[0], preds[0][0], preds[1][0]], ["72","144","288","prediction_144", "prediction_288"])

def residual_blocks(size):
    inp = Input((None,None,size))
    x = Conv2D(32,(3,3),activation=LeakyReLU(0.2), padding='same')(inp)
    x = Conv2D(32,(3,3),activation=LeakyReLU(0.2), padding='same')(x)
    x = Add()([x, inp])
    x = Activation(LeakyReLU(0.2))(x)
    return Model(inp, x)

def create_model_3():
    inp = Input((None,None,3))
    x = Conv2D(32,(3,3),activation=LeakyReLU(0.2), padding='same')(inp)
    x = residual_blocks(32)(x)
    x = residual_blocks(32)(x)
    
    x = UpSampling2D(2)(x)
    x_144 = Conv2D(3,(1,1),activation='sigmoid', padding='same')(x)
    
    x_288 = residual_blocks(32)(x)
    x_288 = UpSampling2D(2)(x)
    x_288 = Conv2D(3, (1,1),activation='sigmoid', padding='same')(x_288)
    
    model = Model(inputs=inp,outputs=[x_144,x_288])
    model.summary()
    return model
model_3 = create_model_3()
model_3.compile(loss='mean_squared_error', optimizer='adam', metrics=[PSNR])

history = model_3.fit(x_72_train, [y_144_train, y_288_train], validation_data=[x_72_validation, [y_144_validation,y_288_validation]], epochs=50)
preds = model_3.predict(x_72_validation)
plot_img([x_72_validation[0], y_144_validation[0], y_288_validation[0], preds[0][0], preds[1][0]], ["72","144","288","prediction_144", "prediction_288"])
def dilated_block (size):
    inp = Input((None,None,size))
    x1 = Conv2D(32, (3, 3), activation=LeakyReLU(0.2), padding='SAME', dilation_rate=1)(inp)
    x2 = Conv2D(32, (3, 3), activation=LeakyReLU(0.2), padding='SAME', dilation_rate=2)(inp)
    x3 = Conv2D(32, (3, 3), activation=LeakyReLU(0.2), padding='SAME', dilation_rate=4)(inp)
    x = Concatenate()([x1, x2,x3])
    x = Activation(LeakyReLU(0.2))(x)
    x = Conv2D(32,(3,3),activation=LeakyReLU(0.2), padding='same')(x)
    return Model(inp, x)

def create_model_4():
    inp = Input((None,None,3))
    x = Conv2D(32,(3,3),activation=LeakyReLU(0.2), padding='same')(inp)
    x = dilated_block(32)(x)
    x = dilated_block(32)(x)
    x = UpSampling2D(2)(x)
    x_144 = Conv2D(3,(1,1),activation='sigmoid', padding='same')(x)
    x_288 = dilated_block(32)(x)
    x_288 = UpSampling2D(2)(x)
    x_288 = Conv2D(3, (1,1),activation='sigmoid', padding='same')(x_288)
    model = Model(inputs=inp,outputs=[x_144,x_288])
    model.summary()
    return model



model_4 = create_model_4()
model_4.compile(loss='mean_squared_error', optimizer='adam', metrics=[PSNR])

history = model_4.fit(x_72_train, [y_144_train, y_288_train], validation_data=[x_72_validation, [y_144_validation,y_288_validation]], epochs=50)
preds = model_4.predict(x_72_validation)
plot_img([x_72_validation[0], y_144_validation[0], y_288_validation[0], preds[0][0], preds[1][0]], ["72","144","288","prediction_144", "prediction_288"])
from keras.applications.vgg16 import VGG16
def create_model_5():
    inp = Input((None,None,3))
    x = Conv2D(64,(3,3),activation=LeakyReLU(0.2), padding='same')(inp)    
    x = Conv2D(64,(3,3),activation=LeakyReLU(0.2), padding='same')(inp)
    vgg_model = VGG16(weights='imagenet', include_top=False, input_shape = (None,None, 3))
    vgg_layer = Model(inputs=vgg_model.input, outputs=vgg_model.get_layer("block1_conv2").output)(inp)
#     model2 = vgg_layer(inp)
    x = Concatenate()([x,vgg_layer])
    
    x = UpSampling2D(2)(x)
    x_144 = Conv2D(3,(1,1),activation='sigmoid', padding='same')(x)
    
    x_288 = UpSampling2D(2)(x_144)
    x_288 = Conv2D(3, (1,1),activation='sigmoid', padding='same')(x_288)
    
    model = Model(inputs=inp,outputs=[x_144,x_288])
    model.summary()
    return model



model_5 = create_model_5()
model_5.compile(loss='mean_squared_error', optimizer='adam', metrics=[PSNR])

history = model_5.fit(x_72_train, [y_144_train, y_288_train], validation_data=[x_72_validation, [y_144_validation,y_288_validation]], epochs=50)
preds = model_5.predict(x_72_validation)
plot_img([x_72_validation[0], y_144_validation[0], y_288_validation[0], preds[0][0], preds[1][0]], ["72","144","288","prediction_144", "prediction_288"])


from keras.layers import Lambda
import tensorflow as tf

def pixel_shuffle(scale=2):
    return lambda x: tf.nn.depth_to_space(x, scale)
def create_model_6():
    vgg_model = VGG16(weights='imagenet', include_top=False, input_shape = (None,None, 3))
    vgg_layer = Model(inputs=vgg_model.input, outputs=vgg_model.get_layer("block1_conv2").output)


    inp = Input((None,None,3))
    x = Conv2D(64,(3,3),activation=LeakyReLU(0.2),padding='SAME')(inp)
    x = Conv2D(64,(3,3),activation=LeakyReLU(0.2),padding='SAME')(x)


    model2 = vgg_layer(inp) 
    x = Concatenate()([x,model2]) 

    # x = UpSampling2D(2)(x)
    x = Lambda(pixel_shuffle())(x)
    med = Conv2D(3, (1,1),activation=LeakyReLU(0.2), padding='SAME')(x)

    # x2 = UpSampling2D(2)(x)
    x2 = Lambda(pixel_shuffle())(x)
    x2 = Conv2D(3, (1,1),activation=LeakyReLU(0.2), padding='SAME')(x2)

    model = Model(inputs=inp,outputs=[med, x2])
    model.summary()
    return model


model_6 = create_model_6()
model_6.compile(loss='mean_squared_error', optimizer='adam', metrics=[PSNR])

history = model_6.fit(x_72_train, [y_144_train, y_288_train], validation_data=[x_72_validation, [y_144_validation,y_288_validation]], epochs=50)
preds = model_6.predict(x_72_validation)
plot_img([x_72_validation[0], y_144_validation[0], y_288_validation[0], preds[0][0], preds[1][0]], ["72","144","288","prediction_144", "prediction_288"])

