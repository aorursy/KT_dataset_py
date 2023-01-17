#__author__ = 'kjeanclaude: https://kaggle.com/kjeanclaude'
# Inspired from this kernel https://www.kaggle.com/keegil/keras-u-net-starter-lb-0-233
import os
import sys
import random
import warnings

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from tqdm import tqdm
from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from scipy.misc import imresize 
from skimage.morphology import label
from PIL import Image

from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K

import tensorflow as tf

%matplotlib inline

# Set some parameters
IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3

# First, we work on the train-test-samples from my (https://www.kaggle.com/kjeanclaude/download-train-test-samples-for-local-prototype) kernel. 
# We could generalize later on the whole competition dataset.
TRAIN_PATH = '../input/download-train-test-samples-for-local-prototype/train/train_color/'
LABEL_PATH = '../input/download-train-test-samples-for-local-prototype/train/train_label/'
TEST_PATH = '../input/download-train-test-samples-for-local-prototype/test/'

#print(os.listdir("../input/"))
warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
seed = 77
random.seed = seed
np.random.seed = seed
print('Done!')

print("tf.__version__ : ", tf.__version__)
print("python --version : ", sys.version)
print("numpy --version : ", np.__version__)
PyVersion = sys.version
!os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Get train and test IDs 
train_ids = next(os.walk(TRAIN_PATH))[2] 
label_ids = next(os.walk(LABEL_PATH))[2] 
test_ids = next(os.walk(TEST_PATH))[2] 
train_ids[:2]
im = imread('../input/download-train-test-samples-for-local-prototype/train/train_color/170927_064455855_Camera_5.jpg', as_grey=False)
im.shape
imshow(im)
plt.show() 
im2 = imread('../input/download-train-test-samples-for-local-prototype/train/train_label/170908_072650121_Camera_5_instanceIds.png', as_grey=True)
im2.shape
from scipy.misc import imresize 
im2 = np.reshape(np.array(im2),(im2.shape[0],im2.shape[1])) 
img = imresize(im2, (IMG_HEIGHT, IMG_WIDTH), mode='L', interp='nearest') 
img = np.reshape(img,(img.shape[0], img.shape[1])) 
img.shape 
imshow(img)
plt.show()

# Get and resize train images and masks
X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)

# dtype=np.float32
features_im_path = []
#features_im_path.append(path) 
print('Getting and resizing train images and masks ... ')
sys.stdout.flush()
for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    path = TRAIN_PATH + id_
    img = imread(path)[:,:,:IMG_CHANNELS]
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_train[n] = img

for n, id_ in tqdm(enumerate(label_ids), total=len(label_ids)):
    path = LABEL_PATH + id_ 
    img = imread(path, as_grey=True) 
    img = np.reshape(np.array(img), (img.shape[0], img.shape[1])) # To transform as an numpy array first
    #img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True) 
    img = imresize(img, (IMG_HEIGHT, IMG_WIDTH), mode='L', interp='nearest') 
    img = np.reshape(img,(img.shape[0], img.shape[1], 1)) 
    Y_train[n] = img 


# Get and resize test images
X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
sizes_test = []
print('Getting and resizing test images ... ')
sys.stdout.flush()
for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
    path = TEST_PATH + id_
    img = imread(path)[:,:,:IMG_CHANNELS]
    sizes_test.append([img.shape[0], img.shape[1]])
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_test[n] = img

print('Done!') 

def display_samples(nb, train_ids):
    #imgs_ids = [c for c in category_to_product.keys() if category_id in str(c)]
    #ix = random.randint(0, len(train_ids))
    ix = random.randint(nb, len(train_ids)-1)
    print('ix value : ', ix)
    imgs_ids = [] 
    imgs_ids.append(X_train[ix]) 
    #imgs_ids.append(X_train[ix]) 
    
    ### For printing purposes, we should reshape it without the last channel
    Y_2 = np.reshape(Y_train[ix],(Y_train[ix].shape[0], Y_train[ix].shape[1])) 
    imgs_ids.append(Y_2) 
    
    print('len imgs_ids : ', len(imgs_ids))
    fig, axs = plt.subplots(1, len(imgs_ids), figsize=(10, 4))
    
    print('axs : ', axs)
    for i, ax_i in enumerate(axs):
        ax_i.imshow(imgs_ids[i])
        ax_i.set_title(i)
        ax_i.grid('off')
        ax_i.axis('off')
        
    #return imgs_ids
display_samples(0, train_ids) 

# Define IoU metric
def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 7)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec))
def mean_iou_(y_pred,y_true):
    y_pred_ = tf.to_int64(y_pred > 0.5)
    y_true_ = tf.to_int64(y_true > 0.5)
    score, up_opt = tf.metrics.mean_iou(y_true_, y_pred_, 7)
    with tf.control_dependencies([up_opt]):
        score = tf.identity(score)
    return score
# Build U-Net model 
inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
s = Lambda(lambda x: x / 255) (inputs)
#s = Lambda(lambda x: x) (inputs)

c1 = Conv2D(8, (3, 3), activation='relu', padding='same') (s)
c1 = Conv2D(8, (3, 3), activation='relu', padding='same') (c1)
p1 = MaxPooling2D((2, 2)) (c1)

c2 = Conv2D(16, (3, 3), activation='relu', padding='same') (p1)
c2 = Conv2D(16, (3, 3), activation='relu', padding='same') (c2)
p2 = MaxPooling2D((2, 2)) (c2)

c3 = Conv2D(32, (3, 3), activation='relu', padding='same') (p2)
c3 = Conv2D(32, (3, 3), activation='relu', padding='same') (c3)
p3 = MaxPooling2D((2, 2)) (c3)

c4 = Conv2D(64, (3, 3), activation='relu', padding='same') (p3)
c4 = Conv2D(64, (3, 3), activation='relu', padding='same') (c4)
p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

c5 = Conv2D(128, (3, 3), activation='relu', padding='same') (p4)
c5 = Conv2D(128, (3, 3), activation='relu', padding='same') (c5)

u6 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c5)
u6 = concatenate([u6, c4])
c6 = Conv2D(64, (3, 3), activation='relu', padding='same') (u6)
c6 = Conv2D(64, (3, 3), activation='relu', padding='same') (c6)

u7 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c6)
u7 = concatenate([u7, c3])
c7 = Conv2D(32, (3, 3), activation='relu', padding='same') (u7)
c7 = Conv2D(32, (3, 3), activation='relu', padding='same') (c7)

u8 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c7)
u8 = concatenate([u8, c2])
c8 = Conv2D(16, (3, 3), activation='relu', padding='same') (u8)
c8 = Conv2D(16, (3, 3), activation='relu', padding='same') (c8)

u9 = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same') (c8)
u9 = concatenate([u9, c1], axis=3)
c9 = Conv2D(8, (3, 3), activation='relu', padding='same') (u9)
c9 = Conv2D(8, (3, 3), activation='relu', padding='same') (c9)

outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)

model = Model(inputs=[inputs], outputs=[outputs]) 
model.compile(optimizer='nadam', loss='binary_crossentropy', metrics=[mean_iou]) 
#model.compile(optimizer='nadam', loss='binary_crossentropy', metrics=['accuracy']) 
model.summary() 

results = model.fit(X_train, Y_train, validation_split=0.1, batch_size=8, epochs=10)

# Predict on train, val and test
#model = load_model('../models/wad-video-seg.h5')
print('Predictions ... ')
preds_train = model.predict(X_train[:int(X_train.shape[0]*0.9)], verbose=1)
preds_val = model.predict(X_train[int(X_train.shape[0]*0.9):], verbose=1)
preds_test = model.predict(X_test, verbose=1)

# Threshold predictions
print('\nPredictions thresholding ... \n')
preds_train_t = (preds_train > 0.5).astype(np.uint8)
preds_val_t = (preds_val > 0.5).astype(np.uint8)
preds_test_t = (preds_test > 0.5).astype(np.uint8)

# Create list of upsampled test masks
print('Test images upsampling ... ')
preds_test_upsampled = []
for i in range(len(preds_test)):
    preds_test_upsampled.append(resize(preds_test[i], 
                                       (sizes_test[i][0], sizes_test[i][1]), 
                                       mode='constant', preserve_range=True))

def display_predictions(nb, preds_train_t):
    #imgs_ids = [c for c in category_to_product.keys() if category_id in str(c)]
    #ix = random.randint(0, len(train_ids))
    ix = random.randint(nb, len(preds_train_t)-1)
    print('ix value : ', ix)
    imgs_ids = [] 
    imgs_labels = []
    imgs_ids.append(X_train[ix]) 
    imgs_labels.append("Image")
    
    ### For printing purposes, we should reshape it without the last channel
    Y_2 = np.reshape(Y_train[ix],(Y_train[ix].shape[0], Y_train[ix].shape[1])) 
    imgs_ids.append(Y_2) 
    imgs_labels.append("Label")
    
    ## train preds
    tpreds = np.reshape(preds_train_t[ix],(preds_train_t[ix].shape[0], preds_train_t[ix].shape[1])) 
    imgs_ids.append(tpreds) 
    imgs_labels.append("Prediction")
    
    print('len imgs_ids : ', len(imgs_ids))
    fig, axs = plt.subplots(1, len(imgs_ids), figsize=(10, 4))
    
    print('axs : ', axs)
    for i, ax_i in enumerate(axs):
        ax_i.imshow(imgs_ids[i])
        ax_i.set_title(imgs_labels[i])
        ax_i.grid('off')
        ax_i.axis('off')
display_predictions(0, preds_train_t) 
display_predictions(0, preds_val_t) 

def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

def prob_to_rles(x, cutoff=0.5): 
    lab_img = label(x > cutoff) 
    for i in range(1, lab_img.max() + 1): 
        yield rle_encoding(lab_img == i) 
# Let's iterate over the test IDs and generate run-length encodings 
# for each seperate mask identified by skimage in the label image ...
new_test_ids = [] 
rles = [] 
for n, id_ in enumerate(test_ids):
    rle = list(prob_to_rles(preds_test_upsampled[n])) 
    #print('rle : ', rle[0])
    #break
    rles.extend(rle) 
    new_test_ids.extend([id_] * len(rle)) 
#rles[0]
sub_sample = pd.read_csv('../input/cvpr-2018-autonomous-driving/sample_submission.csv')
sub_sample.head(2)
# Create submission DataFrame
# The variables order here is relative to the competition #evaluation requirements 
#(different from the 'sample_submission.csv' file)
submission = pd.DataFrame()
submission['ImageId'] = new_test_ids
submission['LabelId'] = 33
submission['Confidence'] = 1
submission['PixelCount'] = 300
submission['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
submission.to_csv('Sub-wad-video-seg.csv', index=False)
submission.head(5)

from IPython.display import FileLink
#%cd $LESSON_HOME_DIR
FileLink('Sub-wad-video-seg.csv')
