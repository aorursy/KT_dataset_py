from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
import os 
import sys 
import random 
import warnings
import numpy as np 
from time import time 
import matplotlib.pyplot as plt 
from tqdm import tqdm
from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label

import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras.layers import Input, Dropout, Lambda, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, Activation, add, multiply, UpSampling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import backend as K 

from sklearn.metrics import classification_report, confusion_matrix
from IPython.display import Image
%matplotlib inline 

IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNEL = 3
TRAIN_PATH = 'E:/Kaggle Dataset/Nuclei/train/'
TEST_PATH = 'E:/Kaggle Dataset/Nuclei/test/'

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
seed = 42
random.seed = seed 
np.random.seed = seed

train_ids = next(os.walk(TRAIN_PATH))[1]
test_ids = next(os.walk(TEST_PATH))[1]
print('Getting and resizing training images ...')
X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNEL), dtype=np.uint8)
Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)

sys.stdout.flush()
for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    path = TRAIN_PATH + id_
    img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNEL]
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_train[n] = img
    mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)

    for mask_file in next(os.walk(path + '/masks/'))[2]:
        mask_ = imread(path + '/masks/' + mask_file)
        mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True), axis=-1)
        
        mask = np.maximum(mask, mask_)

    Y_train[n] = mask

X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNEL), dtype=np.uint8)
size_test = []
print('Getting and resizing test images ...')
sys.stdout.flush()

for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
    path = TEST_PATH + id_
    img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNEL]
    size_test.append([img.shape[0], img.shape[1]])
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_test[n] = img

print('Done')
# Illustrate the train images and masks
plt.figure(figsize=(20, 16))
x, y = 12, 4
for i in range(y):
    for j in range(x):
        plt.subplot(y*2, x, i*2*x+j+1)
        pos = i*120 + j*10
        plt.imshow(X_train[pos])
        plt.title('Image #{}'.format(pos))
        plt.axis('off')
        plt.subplot(y*2, x, (i*2+1)*x+j+1)

        plt.imshow(np.squeeze(Y_train[pos]), cmap='gray_r')
        plt.title('Mask #{}'.format(pos))
        plt.axis('off')

plt.show()
Image(filename="../input/segment/nuclei.JPG", width= 1000, height=1000)
smooth = 1

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)
Image(filename="../input/resunet/model.JPG", width= 350, height=350)
def bn_act(x, act=True):
    x = tf.keras.layers.BatchNormalization()(x)
    if act == True:
        x = tf.keras.layers.Activation("relu")(x)
    return x 

def conv_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    conv = bn_act(x)
    conv = tf.keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(conv)
    return conv 

def stem(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    conv = tf.keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(x)
    conv = conv_block(conv, filters, kernel_size=kernel_size, padding=padding, strides=strides)

    shortcut = tf.keras.layers.Conv2D(filters, kernel_size=(1, 1), padding=padding, strides=strides)(x)
    shortcut = bn_act(shortcut, act=False)

    output = tf.keras.layers.Add()([conv, shortcut])
    return output 

def residual_block(x, filters, kernel_size=(3, 3), padding='same', strides=1):
    res = conv_block(x, filters, kernel_size=kernel_size, padding=padding, strides=strides)
    res = conv_block(res, filters, kernel_size=kernel_size, padding=padding, strides=1)

    shortcut = tf.keras.layers.Conv2D(filters, kernel_size=(1, 1), padding=padding, strides=strides)(x)
    shortcut = bn_act(shortcut, act=False)
    
    output = tf.keras.layers.Add()([shortcut, res])
    return output

def upsample_concat_block(x, xskip):
    u = tf.keras.layers.UpSampling2D((2, 2))(x)
    c = tf.keras.layers.Concatenate()([u, xskip])
    return c 
def ResUNet():
    f = [16, 32, 64, 128, 256]
    inputs = keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNEL))

    ## ENCODER 
    e0 = inputs
    e1 = stem(e0, f[0])
    e2 = residual_block(e1, f[1], strides=2)
    e3 = residual_block(e2, f[2], strides=2)
    e4 = residual_block(e3, f[3], strides=2)
    e5 = residual_block(e4, f[4], strides=2)

    # BRIDGE
    b0 = conv_block(e5, f[4], strides=1)
    b1 = conv_block(b0, f[4], strides=1)

    # DECODER 
    u1 = upsample_concat_block(b1, e4)
    d1 = residual_block(u1, f[4])

    u2 = upsample_concat_block(d1, e3)
    d2 = residual_block(u2, f[3])

    u3 = upsample_concat_block(d2, e2)
    d3 = residual_block(u3, f[2])

    u4 = upsample_concat_block(d3, e1)
    d4 = residual_block(u4, f[1])

    outputs = keras.layers.Conv2D(1, (1, 1), padding='same', activation='sigmoid')(d4)
    model = keras.models.Model(inputs, outputs)
    return model
model = ResUNet()
model.summary()
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=[dice_coef]
)
epochs = 10
model.fit(X_train, Y_train, validation_split=0.1, batch_size=16, epochs=epochs)
# The first 90% used for training
pred_train = model.predict(X_train[:int(X_train.shape[0]*0.9)].astype(np.float16), verbose=1)
# The last 10% used for validation
pred_val = model.predict(X_train[int(X_train.shape[0]*0.9):].astype(np.float16), verbose=1)

# pred_test = model.predict(X_test, verbose=1)

# Thresholds prediction
pred_train_threshold = (pred_train > 0.5).astype(np.float16)
pred_val_threshold = (pred_val > 0.5).astype(np.float16)
## Showing our predicted masks on our training data
ix = random.randint(0, 682)
plt.figure(figsize=(20, 28))

# Our original training image
plt.subplot(131)
imshow(X_train[ix])
plt.title('Image')

# Our original combined mask
plt.subplot(132)
imshow(np.squeeze(Y_train[ix]))
plt.title('Mask')

# The mask of our model U-Net prediction
plt.subplot(133)
imshow(np.squeeze(pred_train_threshold[ix] > 0.5))
plt.title('Prediction')
plt.show()
Image(filename="../input/segment/trainpred.JPG", width= 1000, height=1000)
## Showing our predicted masks on our training data
ix = random.randint(602, 668)
plt.figure(figsize=(20, 28))

# Our original training image
plt.subplot(121)
imshow(X_train[ix])
plt.title('Image')

# The mask of our model U-Net prediction
plt.subplot(122)
ix = ix - 603
imshow(np.squeeze(pred_val_threshold[ix] > 0.5))
plt.title('Prediction')
plt.show()
Image(filename="../input/segment/testpred.JPG", width= 800, height=800)
