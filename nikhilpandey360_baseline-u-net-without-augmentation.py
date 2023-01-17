import numpy as np 
import tensorflow as tf
import pandas as pd
import os
from cv2 import imread, createCLAHE 
import cv2
from glob import glob
%matplotlib inline
import matplotlib.pyplot as plt

image_path = ("../input/pulmonary-chest-xray-abnormalities/Montgomery/MontgomerySet/CXR_png/")
mask_path = ("../input/pulmonary-chest-xray-abnormalities/Montgomery/MontgomerySet/ManualMask/")
rightMaskPath = os.path.join(mask_path,"./rightMask/")
leftMaskPath = os.path.join(mask_path,"./leftMask/")
def getData(image_path,rightMaskPath,leftMaskPath):
    images = os.listdir(image_path)
    rightMask = os.listdir(rightMaskPath)
    leftMask = os.listdir(leftMaskPath)

    print ("Mask Right:",len(rightMask))
    print ("Mask Left:",len(leftMask))
    print ("Image",len(os.listdir(image_path)))
    data = list(set(rightMask) & set(images)  & set(leftMask))
    print("total common image:", len(data))
    
    return data

data = getData(image_path,rightMaskPath,leftMaskPath)
#probably not the best way to read image
file = data[15]

def getClass(fname):
    return int((fname.split(".")[0]).split("_")[-1])


def getMask(file_name):
    l = cv2.imread(leftMaskPath+file_name)
    r = cv2.imread(rightMaskPath+file_name)
    added_image = cv2.addWeighted(l,0.5,r,0.5,0)
    added_image=cv2.threshold(added_image,20,255,cv2.THRESH_BINARY)[1]
    return added_image[:,:,0]


def getImage(file_name):
    lung = cv2.imread(image_path+file_name)
    return lung[:,:,0]
   

lung = getImage(file)
mask = getMask(file)

plt.figure(figsize=(25,10))
plt.subplot(121)
plt.imshow(lung)
plt.subplot(122)
plt.imshow(mask)

plt.show()
lung.shape
# Data Distribution
pos=neg = 0
for i in data:
    clas =getClass(i) 
    if clas:
        pos+=1
    else:
        neg+=1
print("pos neg ::",pos,neg)
# loading data in RAM
from tqdm import tqdm
x_dim = 256
y_dim =256
images = [cv2.resize(getImage(img),(x_dim,y_dim))  for img in tqdm(data)]
masks = [cv2.resize(getMask(img),(x_dim,y_dim))  for img in tqdm(data) ]

images = np.array(images).reshape(len(images),x_dim,x_dim,1)
masks = np.array(masks).reshape(len(images),x_dim,x_dim,1)
plt.imshow(np.squeeze(images[1]))
images[1].shape
# Always do sanity check. Will save you hours during debugging and the urge to stab yourself in brain! (/s)
ls = np.random.randint(1,138,5)


for i in range(3):
    plt.figure(figsize=(25,20))
    plt.subplot(3,1,i+1)
    stacked = np.hstack((np.squeeze(images[i]),np.squeeze(masks[i])))
    plt.imshow(stacked)
# Define a simpe model now to train the images
# source: https://www.kaggle.com/kmader/training-u-net-on-tb-images-to-segment-lungs
from keras.optimizers import Adam
import keras.backend as K
from keras.optimizers import Adam
from keras.losses import binary_crossentropy

reg_param = 1.0
lr = 2e-4
dice_bce_param = 0.0
use_dice = True

def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    return K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)

def dice_p_bce(in_gt, in_pred):
    return dice_bce_param*binary_crossentropy(in_gt, in_pred) - dice_coef(in_gt, in_pred)

def true_positive_rate(y_true, y_pred):
    return K.sum(K.flatten(y_true)*K.flatten(K.round(y_pred)))/K.sum(y_true)


from keras.layers import Conv2D, Activation, Input, UpSampling2D, concatenate, BatchNormalization
from keras.layers import LeakyReLU
from keras.initializers import RandomNormal


def c2(x_in, nf, strides=1):
    x_out = Conv2D(nf, kernel_size=3, padding='same',
                   kernel_initializer='he_normal', strides=strides)(x_in)
    x_out = LeakyReLU(0.2)(x_out)
    return x_out


def unet_enc(vol_size, enc_nf, pre_filter = 8):
    src = Input(shape=vol_size + (1,), name = 'EncoderInput')
    # down-sample path.
    x_in = BatchNormalization(name = 'NormalizeInput')(src)
    x_in = c2(x_in, pre_filter, 1)
    x0 = c2(x_in, enc_nf[0], 2)  
    x1 = c2(x0, enc_nf[1], 2)  
    x2 = c2(x1, enc_nf[2], 2)  
    x3 = c2(x2, enc_nf[3], 2) 
    return Model(inputs = [src], 
                outputs = [x_in, x0, x1, x2, x3],
                name = 'UnetEncoder')

from keras.models import Model
from keras import layers
def unet(vol_size, enc_nf, dec_nf, full_size=True, edge_crop=48):
    """
    unet network for voxelmorph 
    Args:
        vol_size: volume size. e.g. (256, 256, 256)
        enc_nf: encoder filters. right now it needs to be to 1x4.
            e.g. [16,32,32,32]
            TODO: make this flexible.
        dec_nf: encoder filters. right now it's forced to be 1x7.
            e.g. [32,32,32,32,8,8,3]
            TODO: make this flexible.
        full_size
    """

    # inputs
    raw_src = Input(shape=vol_size + (1,), name = 'ImageInput')
    src = layers.GaussianNoise(0.25)(raw_src)
    enc_model = unet_enc(vol_size, enc_nf)
    # run the same encoder on the source and the target and concatenate the output at each level
    x_in, x0, x1, x2, x3 = [s_enc for s_enc in enc_model(src)]

    x = c2(x3, dec_nf[0])
    x = UpSampling2D()(x)
    x = concatenate([x, x2])
    x = c2(x, dec_nf[1])
    x = UpSampling2D()(x)
    x = concatenate([x, x1])
    x = c2(x, dec_nf[2])
    x = UpSampling2D()(x)
    x = concatenate([x, x0])
    x = c2(x, dec_nf[3])
    x = c2(x, dec_nf[4])
    x = UpSampling2D()(x)
    x = concatenate([x, x_in])
    x = c2(x, dec_nf[5])

    # transform the results into a flow.
    y_seg = Conv2D(1, kernel_size=3, padding='same', name='lungs', activation='sigmoid')(x)
    y_seg = layers.Cropping2D((edge_crop, edge_crop))(y_seg)
    y_seg = layers.ZeroPadding2D((edge_crop, edge_crop))(y_seg)
    # prepare model
    model = Model(inputs=[raw_src], outputs=[y_seg])
    return model

# https://www.kaggle.com/krishanudb/keras-based-unet-model-construction-tutorial
# use the predefined depths
nf_enc=[16,32,32,32]
nf_dec=[32,32,32,32,32,16,16,2]
OUT_DIM=(256,256)
net = unet(OUT_DIM, nf_enc, nf_dec)
# ensure the model roughly works
a= net.predict([np.zeros((1,)+OUT_DIM+(1,))])
print(a.shape)
net.summary()
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
weight_path="{}_weights.best.hdf5".format('cxr_reg')

checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, 
                             save_best_only=True, mode='min', save_weights_only = True)

reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.5, 
                                   patience=3, 
                                   verbose=1, mode='min', epsilon=0.0001, cooldown=2, min_lr=1e-6)
early = EarlyStopping(monitor="val_loss", 
                      mode="min", 
                      patience=15) # probably needs to be more patient, but kaggle time is limited
callbacks_list = [checkpoint, early, reduceLROnPlat]
from IPython.display import clear_output
from keras.optimizers import Adam 
from sklearn.model_selection import train_test_split

net.compile(optimizer=Adam(lr=lr), 
              loss=[dice_p_bce], 
           metrics = [true_positive_rate, 'binary_accuracy'])

train_vol, test_vol, train_seg, test_seg = train_test_split((images-127.0)/127.0, 
                                                            (masks>127).astype(np.float32), 
                                                            test_size = 0.2, 
                                                            random_state = 2018)
loss_history = net.fit(x = train_vol,
                       y = train_seg,
                       
                  epochs = 100,
                  validation_data =(test_vol,test_seg) ,
                  callbacks=callbacks_list
                 )
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (10, 5))
ax1.plot(loss_history.history['loss'], '-', label = 'Loss')
ax1.plot(loss_history.history['val_loss'], '-', label = 'Validation Loss')
ax1.legend()

ax2.plot(100*np.array(loss_history.history['binary_accuracy']), '-', 
         label = 'Accuracy')
ax2.plot(100*np.array(loss_history.history['val_binary_accuracy']), '-',
         label = 'Validation Accuracy')
ax2.legend()
prediction = net.predict(images[1:10])
plt.figure(figsize=(20,10))

for i in range(0,6,3):
    plt.subplot(2,3,i+1)
    plt.imshow(np.squeeze(images[i]))
    plt.xlabel("Base Image")
    
    plt.subplot(2,3,i+2)
    plt.imshow(np.squeeze(prediction[i]))
    plt.xlabel("Pridiction")
    plt.subplot(2,3,i+3)
    plt.imshow(np.squeeze(masks[i]))
    plt.xlabel("Segmentation Ground Truth")