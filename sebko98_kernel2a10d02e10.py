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
import tensorflow as tf
import os
import sys
import cv2
 
import numpy as np
 
from tqdm import tqdm
from itertools import chain
 
from skimage.io import imread, imshow
from skimage.transform import resize
from skimage import img_as_bool
import random
import matplotlib.pyplot as plt
import time
import cv2
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()

def load_data(frames, masks,IMG_WIDTH=1280,IMG_HEIGHT=1024,IMG_CHANNELS=3,seed=42):
    
#     random.seed = seed
#     np.random.seed = seed
    
    X = np.zeros((len(os.listdir(frames) ), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.uint8)
    Y = np.zeros((len(os.listdir(frames) ), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)

    n = sorted(os.listdir(frames)) 
    o = sorted(os.listdir(masks))
  
    for i in range(0,len(n)):
        img = cv2.imread(frames +'/'+n[i])[:,:,:IMG_CHANNELS]
        img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        X[i] = img
        mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool) 
        mask_ = cv2.imread(masks +'/'+o[i],0)
       # print(mask_)
#         mask_ = (mask_ == 6)
        mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True), axis=-1)
        mask = np.maximum(mask, mask_)
        Y[i] = mask
    return X,Y


def load_data_by_class(frames, masks, classes, IMG_WIDTH=1280,IMG_HEIGHT=1024,IMG_CHANNELS=3,seed=42):
    
#     random.seed = seed
#     np.random.seed = seed
    
    X = np.zeros((len(os.listdir(frames) ), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.uint8)
    Y = np.zeros((len(os.listdir(frames) ), IMG_HEIGHT, IMG_WIDTH, len(classes)), dtype=np.uint8)

    
    n = sorted(os.listdir(frames))
    o = sorted(os.listdir(masks))

    for i in tqdm(range(0,len(n))):
#         print(i)
        img = cv2.imread(frames +'/'+n[i])[:,:,:IMG_CHANNELS]
        img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        img = resize(img, (IMG_HEIGHT, IMG_WIDTH,1), mode='constant', preserve_range=True)
        X[i] = img

        mask_ = cv2.imread(masks +'/'+o[i],0)
        masks_ = [(mask_ == c).squeeze() for c in classes]
        masks_ = [resize(m, (IMG_HEIGHT, IMG_WIDTH),anti_aliasing=False) for m in masks_]
        mask = np.stack(masks_, axis=-1).astype('bool')
        Y[i] = mask
    
    return X,Y


def get_data(DATA_PATH,X_PATH,Y_PATH,classes):
    folders = next(os.walk(DATA_PATH))[1]

    x = []
    y = []
    for folder in folders:
        if folder == 'city':
            print('skip')
            continue
        DATA = os.path.join(DATA_PATH,folder)
#         print(DATA)
        COMPLETE_X_PATH = os.path.join(DATA,X_PATH)
#         print(COMPLETE_X_PATH)
        COMPLETE_Y_PATH = os.path.join(DATA,Y_PATH)

        frames, masks = load_data_by_class(COMPLETE_X_PATH,COMPLETE_Y_PATH,classes)
        if len(x) == 0:
            x = frames
            y = masks
        else:
            x = np.concatenate((x ,frames),axis=0)
            y = np.concatenate((y, masks),axis=0)
    return x,y

sky = [23]
DATA_PATH = '/kaggle/input/mergeddata/gelabeld/'

x_train, y_train = get_data(DATA_PATH,'train_frames/','train_masks/',sky)
x_val, y_val = get_data(DATA_PATH,'val_frames/','val_masks/',sky)
# Build U-Net model


def unet(IMG_HEIGHT=1024,IMG_WIDTH=1280,IMG_CHANNELS=1,num_classes=1):

    inputs = tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)

    c1 = tf.keras.layers.Conv2D(16, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                                padding='same')(s)
    c1 = tf.keras.layers.Dropout(0.1)(c1)
    c1 = tf.keras.layers.Conv2D(16, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                                padding='same')(c1)
    p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

    c2 = tf.keras.layers.Conv2D(32, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                                padding='same')(p1)
    c2 = tf.keras.layers.Dropout(0.1)(c2)
    c2 = tf.keras.layers.Conv2D(32, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                                padding='same')(c2)
    p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)

    c3 = tf.keras.layers.Conv2D(64, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                                padding='same')(p2)
    c3 = tf.keras.layers.Dropout(0.2)(c3)
    c3 = tf.keras.layers.Conv2D(64, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                                padding='same')(c3)
    p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)

    c4 = tf.keras.layers.Conv2D(128, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                                padding='same')(p3)
    c4 = tf.keras.layers.Dropout(0.2)(c4)
    c4 = tf.keras.layers.Conv2D(128, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                                padding='same')(c4)
    p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = tf.keras.layers.Conv2D(256, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                                padding='same')(p4)
    c5 = tf.keras.layers.Dropout(0.3)(c5)
    c5 = tf.keras.layers.Conv2D(256, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                                padding='same')(c5)

    u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = tf.keras.layers.concatenate([u6, c4])
    c6 = tf.keras.layers.Conv2D(128, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                                padding='same')(u6)
    c6 = tf.keras.layers.Dropout(0.2)(c6)
    c6 = tf.keras.layers.Conv2D(128, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                                padding='same')(c6)

    u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = tf.keras.layers.concatenate([u7, c3])
    c7 = tf.keras.layers.Conv2D(64, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                                padding='same')(u7)
    c7 = tf.keras.layers.Dropout(0.2)(c7)
    c7 = tf.keras.layers.Conv2D(64, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                                padding='same')(c7)

    u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = tf.keras.layers.concatenate([u8, c2])
    c8 = tf.keras.layers.Conv2D(32, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                                padding='same')(u8)
    c8 = tf.keras.layers.Dropout(0.1)(c8)
    c8 = tf.keras.layers.Conv2D(32, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                                padding='same')(c8)

    u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
    c9 = tf.keras.layers.Conv2D(16, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                                padding='same')(u9)
    c9 = tf.keras.layers.Dropout(0.1)(c9)
    c9 = tf.keras.layers.Conv2D(16, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                                padding='same')(c9)
    activation_func = 'sigmoid' if num_classes==1 else 'softmax'
    outputs = tf.keras.layers.Conv2D(num_classes, (1, 1), activation=activation_func)(c9) #softmax for multi class

    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    
    model.summary()
    return model

def shallow_unet(IMG_HEIGHT=1024,IMG_WIDTH=1280,IMG_CHANNELS=1,num_classes=1):
    inputs = tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)

    c1 = tf.keras.layers.Conv2D(16, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                                padding='same')(s)
    c1 = tf.keras.layers.Dropout(0.1)(c1)

    c1 = tf.keras.layers.Conv2D(16, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                                padding='same')(c1)
    
    p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

    c2 = tf.keras.layers.Conv2D(32, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                                padding='same')(p1)
    c2 = tf.keras.layers.Dropout(0.2)(c2)

    c2 = tf.keras.layers.Conv2D(32, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                                padding='same')(c2)
    u2 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c2)

    u3 = tf.keras.layers.concatenate([u2, c1])

    
    c3 = tf.keras.layers.Conv2D(16, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                                padding='same')(u3)
    c3 = tf.keras.layers.Dropout(0.1)(c3)     
    c3 = tf.keras.layers.Conv2D(32, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                                padding='same')(c3)

    activation_func = 'sigmoid' if num_classes==1 else 'softmax'
    outputs = tf.keras.layers.Conv2D(num_classes, (1, 1), activation=activation_func)(c3) #softmax for multi class

    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    
    model.summary()
    return model
from tensorflow.keras import backend as K
def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.dot(y_true, K.transpose(y_pred))
    union = K.dot(y_true,K.transpose(y_true))+K.dot(y_pred,K.transpose(y_pred))
    return (2. * intersection + smooth) / (union + smooth)

def dice_coef_loss(y_true, y_pred):
    return K.mean(1-dice_coef(y_true, y_pred),axis=-1)

def dice_coef_binary(y_true, y_pred, smooth=1e-7):
    '''
    Dice coefficient for 2 categories. Ignores background pixel label 0
    Pass to model as metric during compile statement
    '''
    y_true_f = K.flatten(K.one_hot(K.cast(y_true, 'int32'), num_classes=2)[...,1:])
    y_pred_f = K.flatten(y_pred[...,1:])
    intersect = K.sum(y_true_f * y_pred_f, axis=-1)
    denom = K.sum(y_true_f + y_pred_f, axis=-1)
    return K.mean((2. * intersect / (denom + smooth)))


def dice_coef_binary_loss(y_true, y_pred):
    '''
    Dice loss to minimize. Pass to model as loss during compile statement
    '''
    return 1 - dice_coef_binary(y_true, y_pred)
try:
    os.makedirs('../myoutputs')
except:
    pass
from datetime import datetime
now = datetime.now()
date_time = now.strftime("%m-%d-%Y_%H-%M")
print("date and time:",date_time)
model_dir = '../myoutputs/sky-net_'+date_time+'.h5'
weights_path = '../myoutputs/weights_'+date_time+'.hdf5'
os.path.exists('../myoutputs/')
from tensorflow.keras import optimizers
adam = optimizers.Adam(lr=0.0001)
model = unet(IMG_HEIGHT=1024,IMG_WIDTH=1280,IMG_CHANNELS=1,num_classes=1)
model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['acc']) # categorical cross entropy

# from IPython.display import SVG
# from tensorflow.keras.utils import model_to_dot

# SVG(model_to_dot(model).create(prog='dot', format='svg'))

from tensorflow.keras.utils import plot_model
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True) 

from tensorflow.keras.callbacks import ModelCheckpoint


checkpoint_weight = ModelCheckpoint('../myoutputs/weights-val_acc{val_acc:.3f}_'+date_time+'.hdf5', monitor='val_acc', 
                             verbose=1, save_best_only=True, mode='max')

csv_logger = tf.keras.callbacks.CSVLogger('./log.out', append=True, separator=';')

earlystopping = tf.keras.callbacks.EarlyStopping(monitor = 'acc', verbose = 1,
                              min_delta = 0.01, patience = 15, mode = 'max')

my_callbacks = [
    checkpoint_weight, 
    csv_logger, 
    earlystopping,
    tf.keras.callbacks.TensorBoard(log_dir='./logs'), 
]

# aug = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=5, zoom_range=0.15,
#     width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,# brightness_range=[-10,10],
#     horizontal_flip=True)
 
# gen = tf.keras.preprocessing.image.ImageDataGenerator()
# results = model.fit_generator(gen.flow(x_train, y_train, batch_size=16),
#     validation_data=(x_val, y_val), steps_per_epoch=len(x_train) // 16,
#     epochs=50,callbacks=my_callbacks)

results = model.fit(x_train, y_train, batch_size=16, validation_data=(x_val,y_val), shuffle=True, verbose=1, epochs=50, callbacks=my_callbacks)



model.save(model_dir)     
plt.figure(figsize=(8, 8))
plt.title("Learning curve")
plt.plot(results.history["loss"], label="loss")
plt.plot(results.history["val_loss"], label="val_loss")
plt.plot( np.argmin(results.history["val_loss"]), np.min(results.history["val_loss"]), marker="x", color="r", label="best model")
plt.xlabel("Epochs")
plt.ylabel("log_loss")
plt.legend(); 

import tensorflow as tf

# model_dir = '/kaggle/input/trainedmodel/sky-net_11-03-2019_18-36.h5'
weights_path = '../myoutputs/weights-vall_acc0.949_11-09-2019_18-59.hdf5'
model = tf.keras.models.load_model(model_dir)
model.load_weights(weights_path)

x_test=x_train
y_test=y_train

for idx in range(0,len(x_test)):
# idx = random.randint(0, len(x_test))
    x=np.array(x_test[idx])
    x=np.expand_dims(x, axis=0)
    start_predict = time.time()
    predict = model.predict(x, verbose=1)
    print("Time for predeiction:",str(time.time()-start_predict))

#     predict = (predict > 0.5).astype(np.uint8)
    visualize(
            image=cv2.cvtColor(x_test[idx],cv2.COLOR_GRAY2RGB),
            gt_mask=1-y_test[idx].squeeze(),
            pr_mask=np.squeeze(1-predict[0]),
            error = abs((y_test[idx].squeeze().astype(int) - np.squeeze(1-predict[0]))),
            result = cv2.cvtColor(cv2.multiply(1-predict[0],x_test[idx]),cv2.COLOR_GRAY2RGB)
        )

