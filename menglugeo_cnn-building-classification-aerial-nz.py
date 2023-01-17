from tensorflow.keras import backend as K

from tensorflow.keras.optimizers import Adam 

from tensorflow.keras.models import Model, model_from_json, load_model

from tensorflow.keras import optimizers, layers

from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint,LearningRateScheduler, ReduceLROnPlateau, CSVLogger, EarlyStopping

from tensorflow.keras.layers import (

    Conv2D,

    UpSampling2D,

    MaxPooling2D,

    Input,

    Conv2DTranspose,

    Flatten,

    BatchNormalization,

    Activation,

    Concatenate,

    concatenate 

 

)

from tensorflow.keras.layers import RepeatVector, Reshape

from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

from tensorflow.keras.applications import MobileNetV2



from scipy import interpolate

from osgeo import gdal_array

from pathlib import Path

from functools import partial

from sklearn.metrics import jaccard_score

import pandas as pd

import json

import matplotlib.pyplot as plt

import numpy as np

import time

import os

import tensorflow as tf

!pip install keras-unet

!pip install imgaug

import imgaug.augmenters as iaa

from keras_unet.utils import plot_imgs
batch_size = 16

epochs = 10

version = 1

data_augmentation = True

model_type = 'UNet%d' % (version)

translearn = True
seq = iaa.Sequential([

    iaa.Clouds(),

    iaa.Rain(), 

    iaa.Snowflakes(flake_size=(0.2, 0.7), speed=(0.007, 0.03)),

    iaa.GaussianBlur(sigma=(0, 3.0)),

    iaa.AveragePooling(2)

])



def lr_schedule(epoch):

    """Learning Rate Schedule



    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.

    Called automatically every epoch as part of callbacks during training.



    # Arguments

        epoch (int): The number of epochs



    # Returns

        lr (float32): learning rate

    """

    lr = 1e-3

    if epoch > 180:

        lr *= 1e-3

    elif epoch > 160:

        lr *= 5e-2

    elif epoch > 120:

        lr *= 1e-1

    elif epoch > 80:

        lr *= 5e-1

    print('Learning rate: ', lr)

    return lr
def m_u_net(input_shape):

    inputs = Input(shape=input_shape, name="input_image")

    

    encoder = MobileNetV2(input_tensor=inputs, weights="imagenet", include_top=False, alpha=1.3)

    #encoder.trainable=False

    skip_connection_names = ["input_image", "block_1_expand_relu", "block_3_expand_relu", "block_6_expand_relu"]

    encoder_output = encoder.get_layer("block_13_expand_relu").output

    

    f = [16, 32, 48, 64]

    x = encoder_output

    for i in range(1, len(skip_connection_names)+1, 1):

        x_skip = encoder.get_layer(skip_connection_names[-i]).output

        x = UpSampling2D((2, 2))(x)

        x = Concatenate()([x, x_skip])

        

        x = Conv2D(f[-i], (3, 3), padding="same")(x)

        x = BatchNormalization()(x)

        x = Activation("relu")(x)

        

        x = Conv2D(f[-i], (3, 3), padding="same")(x)

        x = BatchNormalization()(x)

        x = Activation("relu")(x)

        

    x = Conv2D(1, (1, 1), padding="same")(x)

    x = Activation("sigmoid")(x)

    

    model = Model(inputs, x)

    return model

def load_rasters(path, pathX, pathY ):  # Subset from original raster with extent and upperleft coord

    """Load training data pairs (two high resolution images and two low resolution images)"""

    pathXabs = os.path.join(path, pathX)

    pathYabs = os.path.join(path, pathY)

    le = len(os.listdir(pathXabs) )

        

    stackX = []

    stackY = []

    for i in range(0, le):

        fileX = os.path.join(pathXabs, os.listdir(pathXabs)[i])

        fileY = os.path.join(pathYabs, os.listdir(pathXabs)[i])

        dataX = gdal_array.LoadFile(fileX) #.astype(np.int),ysize=extent[1],xsize=extent[0]

        stackX.append(dataX)

        dataY = gdal_array.LoadFile(fileY) #.astype(np.int),ysize=extent[1],xsize=extent[0]

        stackY.append(dataY)

    stackX = np.array(stackX)

    stackY = np.array(stackY)

    return stackX, stackY 

X, Y= load_rasters('../input/nzaerial/val','image','label')
np.median(X[1,0,:,:]-X[1,1,:,:] )

np.median(Y)
X_cl = np.moveaxis(X, 1, -1) # channel last: ! be super careful about what array reshape mean, it is not the same as moving axis!!

X_train = X_cl[:int(X_cl.shape[0]*0.8),]

Y_train = Y[:int(Y.shape[0]*0.8),]

X_test = X_cl[int(X_cl.shape[0]*0.8)+1:,] 

Y_test = Y[int(Y.shape[0]*0.8)+1:,] 
seq.show_grid([X_train[0], X_train[1]], cols=8, rows=8)
#model=load_model(root_path+"/models/dense_image_matching_network.hdf5", custom_objects={'mycrossentropy': mycrossentropy})



    #test stage

X2, Y2 = load_rasters('../input/globalcities/Satellite dataset ó± (global cities)','image','label')

X2_cl = np.moveaxis(X2, 1, -1)

Y2_cl = np.moveaxis(Y2, 1, -1)# channel last: ! be super careful about what array reshape mean, it is not the same as moving axis!!

Y2_cl =  Y2_cl[:,:,:,0]
plt.imshow(Y_train[0,:,:])

plt.show()  



plt.imshow(X_test[0,:,:])

plt.show() 

 

plt.imshow(Y_test[0,:,:])

plt.show()  
plot_imgs(

    org_imgs=X_test, # required - original images

    mask_imgs=Y_test , # required - ground truth masks

    

    nm_img_to_plot=3) # optional - number of images to plot
print(X_train.shape, X_test.shape,Y_train.shape, Y_test.shape)
def iou_coef(y_true, y_pred, smooth=1):

  intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])

  union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])-intersection

  iou = K.mean((intersection + smooth) / (union + smooth), axis=0)

  return iou



def iou(y_true, y_pred, smooth=1.):

    y_true_f = K.flatten(y_true)

    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f)

    return (intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + smooth)



def iou2(y_true, y_pred, smooth=np.float32(1.0) ):

    y_pred = K.cast(K.greater(y_pred, .5), dtype='float32')

    y_true = K.cast(y_true, dtype='float32')

    y_true_f = K.flatten(y_true)

    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f)

    return (intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + smooth)



def mean_iou(y_true, y_pred):

    # Consider prediction greater than 0.5

    y_pred = K.cast(K.greater(y_pred, .5), dtype='float32') # .5 is the threshold

    inter = K.sum(K.sum(K.squeeze(y_true * y_pred, axis=3), axis=2), axis=1)

    union = K.sum(K.sum(K.squeeze(y_true + y_pred, axis=3), axis=2), axis=1) - inter

    return K.mean((inter + K.epsilon()) / (union + K.epsilon()))





# Covariance

def cov(y_true, y_pred):

    return K.mean((y_true - K.mean(y_true)) * K.transpose((y_pred - K.mean(y_pred))))





# Correlation

def r2(y_true, y_pred):

    # mean calls tensor property instead of ndarray

    tf_true = y_true

    if not isinstance(y_true, tf.Tensor):

        tf_true = tf.convert_to_tensor(y_true)

    res = K.sum(K.square(y_true - y_pred))

    tot = K.sum(K.square(y_true - K.mean(tf_true)))

    return 1 - res / (tot + K.epsilon())





# Signal-to-noise ratio

def psnr(y_true, y_pred, data_range=50):

    #Peak signal-to-noise ratio averaged over samples and channels

    mse = K.mean(K.square(y_true - y_pred), axis=(-3, -2))

    return K.mean(20 * K.log(data_range / K.sqrt(mse)) / np.log(10))





# structural similarity measurement system

def ssim(y_true, y_pred, data_range=50):

    """structural similarity measurement system."""

    K1 = 0.01

    K2 = 0.03



    mu_x = K.mean(y_pred)

    mu_y = K.mean(y_true)



    sig_x = K.std(y_pred)

    sig_y = K.std(y_true)

    sig_xy = cov(y_true, y_pred)



    L = data_range

    C1 = (K1 * L) ** 2

    C2 = (K2 * L) ** 2



    return ((2 * mu_x * mu_y + C1) * (2 * sig_xy * C2) /

            (mu_x ** 2 + mu_y ** 2 + C1) * (sig_x ** 2 + sig_y ** 2 + C2))



##############################################

# Unet

##############################################

 

def unet(input_shape):

    inputs = Input(shape=input_shape)

    conv1_1 = Conv2D(16, (3, 3), padding='same')(inputs)

    bn1_1 = BatchNormalization(axis=3)(conv1_1)

    relu1_1 = Activation('relu')(bn1_1)

    conv1_2 = Conv2D(16, (3, 3), padding='same')(relu1_1)

    bn1_2 = BatchNormalization(axis=3)(conv1_2)

    relu1_2 = Activation('relu')(bn1_2)

    pool1 = MaxPooling2D(pool_size=(2, 2))(relu1_2)

    

    conv2_1 = Conv2D(32, (3, 3), padding='same')(pool1)

    bn2_1 = BatchNormalization(axis=3)(conv2_1)

    relu2_1 = Activation('relu')(bn2_1)

    conv2_2 = Conv2D(32, (3, 3), padding='same')(relu2_1)

    bn2_2 = BatchNormalization(axis=3)(conv2_2)

    relu2_2 = Activation('relu')(bn2_2)

    pool2 = MaxPooling2D(pool_size=(2, 2))(relu2_2)

    

    conv3_1 = Conv2D(64, (3, 3), padding='same')(pool2)

    bn3_1 = BatchNormalization(axis=3)(conv3_1)

    relu3_1 = Activation('relu')(bn3_1)

    conv3_2 = Conv2D(64, (3, 3), padding='same')(relu3_1)

    bn3_2 = BatchNormalization(axis=3)(conv3_2)

    relu3_2 = Activation('relu')(bn3_2)

    pool3 = MaxPooling2D(pool_size=(2, 2))(relu3_2)

    

    conv4_1 = Conv2D(128, (3, 3), padding='same')(pool3)

    bn4_1 = BatchNormalization(axis=3)(conv4_1)

    relu4_1 = Activation('relu')(bn4_1)

    conv4_2 = Conv2D(128, (3, 3), padding='same')(relu4_1)

    bn4_2 = BatchNormalization(axis=3)(conv4_2)

    relu4_2 = Activation('relu')(bn4_2)

    pool4 = MaxPooling2D(pool_size=(2, 2))(relu4_2)

    

    conv5_1 = Conv2D(256, (3, 3), padding='same')(pool4)

    bn5_1 = BatchNormalization(axis=3)(conv5_1)

    relu5_1 = Activation('relu')(bn5_1)

    conv5_2 = Conv2D(256, (3, 3), padding='same')(relu5_1)

    bn5_2 = BatchNormalization(axis=3)(conv5_2)

    relu5_2 = Activation('relu')(bn5_2)

    

    up6 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(relu5_2), relu4_2], axis=3)

    conv6_1 = Conv2D(128, (3, 3), padding='same')(up6)

    bn6_1 = BatchNormalization(axis=3)(conv6_1)

    relu6_1 = Activation('relu')(bn6_1)

    conv6_2 = Conv2D(128, (3, 3), padding='same')(relu6_1)

    bn6_2 = BatchNormalization(axis=3)(conv6_2)

    relu6_2 = Activation('relu')(bn6_2)

    

    up7 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(relu6_2), relu3_2], axis=3)

    conv7_1 = Conv2D(64, (3, 3), padding='same')(up7)

    bn7_1 = BatchNormalization(axis=3)(conv7_1)

    relu7_1 = Activation('relu')(bn7_1)

    conv7_2 = Conv2D(64, (3, 3), padding='same')(relu7_1)

    bn7_2 = BatchNormalization(axis=3)(conv7_2)

    relu7_2 = Activation('relu')(bn7_2)

    

    up8 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(relu7_2), relu2_2], axis=3)

    conv8_1 = Conv2D(32, (3, 3), padding='same')(up8)

    bn8_1 = BatchNormalization(axis=3)(conv8_1)

    relu8_1 = Activation('relu')(bn8_1)

    conv8_2 = Conv2D(32, (3, 3), padding='same')(relu8_1)

    bn8_2 = BatchNormalization(axis=3)(conv8_2)

    relu8_2 = Activation('relu')(bn8_2)

    

    up9 = concatenate([Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(relu8_2), relu1_2], axis=3)

    conv9_1 = Conv2D(16, (3, 3), padding='same')(up9)

    bn9_1 = BatchNormalization(axis=3)(conv9_1)

    relu9_1 = Activation('relu')(bn9_1)

    conv9_2 = Conv2D(16, (3, 3), padding='same')(relu9_1)

    bn9_2 = BatchNormalization(axis=3)(conv9_2)

    relu9_2 = Activation('relu')(bn9_2)

    

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(relu9_2)

    

    model = Model(inputs=[inputs], outputs=[conv10])

    print(model.summary())

    

    return model





def get_model(name):

    """Get model function from the name space in strings"""

    return globals()[name]

X_train.shape[1:]
if version == 1:

    unet_model =unet(input_shape=X_train.shape[1:])

else:

    unet_model =m_u_net(input_shape=X_train.shape[1:])

unet_model.compile(loss='binary_crossentropy', metrics=[iou2],

              optimizer=Adam(learning_rate=lr_schedule(0))) 

tf.keras.utils.plot_model(unet_model, to_file="/kaggle/working/model_cnn.png", show_shapes=True)

#[iou]


# Prepare model model saving directory.

save_dir = os.path.join('saved_models')

model_name = 'mix_%s_model.{epoch:03d}.h5' % model_type  

if not os.path.isdir(save_dir):

    os.makedirs(save_dir)

filepath = os.path.join(save_dir, model_name)



# Prepare callbacks for model saving and for learning rate adjustment.

checkpoint = ModelCheckpoint(filepath=filepath,

                             monitor='val_acc',

                             verbose=1,

                             save_best_only=True)



lr_scheduler = LearningRateScheduler(lr_schedule)



lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),

                               cooldown=0,

                               patience=5,

                               min_lr=0.5e-6)



callbacks = [checkpoint, lr_reducer, lr_scheduler]
 

# Run training, with or without data augmentation.

if  data_augmentation:

    print('using data augmentation.')

    X_train = seq(images=X_train)

    



history = unet_model.fit(X_train, Y_train,batch_size=batch_size,

              epochs=epochs,

              validation_data=(X_test, Y_test),

                    callbacks=callbacks)

 

# Score trained model.

scores =unet_model.evaluate(X_test , Y_test, verbose=1)

print('Test loss:', scores[0])

print('Test accuracy:', scores[1])
plot_imgs(

    org_imgs=X_train[ :,:,:],  

    mask_imgs=Y_train[:,:,: ],   

    nm_img_to_plot=2) # optional - number of images to plot
print(X_train.shape, X_test.shape,Y_train.shape, Y_test.shape)
unet_model.save('NZunet.h5')
 

plt.plot(history.history['iou2'])

plt.plot(history.history['val_iou2'])

plt.title('model accuracy')

plt.ylabel('iou2')

plt.xlabel('epoch')

plt.legend(["train_iou2", "val_iou2"], loc='upper left')

# save before show(), otherwise wont work

 

plt.show()
if translearn == True:

    X_test = X2_cl

    Y_test = Y2_cl

testpre = unet_model.predict(X_test)
Y_test=np.where(Y_test > 0 , 1., 0)
 

unet_model = load_model('./NZunet.h5',compile = False)

unet_model.compile(loss='binary_crossentropy', metrics=[iou2],

              optimizer=Adam(learning_rate=lr_schedule(0))) 

score = unet_model.evaluate(X_test, Y_test, verbose=0)

print ("%s: %.2f%%" % (unet_model.metrics_names[1], score[1]*100))

plot_imgs(

    org_imgs=X_test[ :,:,:],  

    mask_imgs=Y_test[:,:,: ],  

    pred_imgs=testpre[:,:,: ],

    nm_img_to_plot=40) # optional - number of images to plot
X_tr2 = X_test[0:100,]

Y_tr2 = Y_test[0:100,]

X_te2 = X_test[101:,]

Y_te2 = Y_test[101:,]
model = unet_model

model = Model(inputs=model.inputs, outputs=model.layers[-2].output)



unet_model.trainable = False
X_tr2[1: ].shape
plot_imgs(

    org_imgs=X_test[ :,:,:],  

    mask_imgs=Y_test[:,:,: ],  

    pred_imgs=testpre2[:,:,: ],

    nm_img_to_plot=20) # optional - number of images to plot
def iou(Y_test, testpre, smooth=np.float32(1.0)):

    inter = K.sum(K.flatten(Y_test)*K.flatten(testpre))+smooth

    union = K.sum(K.flatten(Y_test))+K.sum(K.flatten(testpre))+smooth - inter

    iou = inter /union 

    return tf.keras.backend.get_value(iou)
iou(Y_test , testpre2)
import inspect

inspect.getargspec(plot_imgs)