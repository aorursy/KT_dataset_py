# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import glob
import random
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import cv2
import imgaug as ia
import imgaug.augmenters as iaa
import os
print(os.listdir("../input"))
import tensorflow as tf
# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/train_labels.csv')
df.head()
df_train_paths = glob.glob('../input/train/*.tif')
df_test_paths = glob.glob('../input/test/*.tif')
df_train_paths[:5]
df_test_paths[:5]
df['label'].value_counts()
def label_mapping(data):
    return data.split('/')[-1].replace('.tif','')
id_label = {k:v for k,v in zip(df.id.values,df.label.values)}
def give_label(img_path):
    return id_label[img_path]
def get_batch(data, batch_size):
    return (data[i:i+batch_size] for i in range(0, len(data), batch_size))   
def data_aug(img_data,train_label_mapping,batch_size,augment = False):
    seq = get_seq()
    while True:
        random.shuffle(img_data)
        for batch in get_batch(img_data,batch_size):
            X = [cv2.imread(img_path) for img_path in batch]
            y = [train_label_mapping[get_id_from_img_path(img_path)] for img_path in batch]
            
            if augment:
                X = seq.augment_images(X)
    
            yield np.array(X), np.array(y)
# df_main.head()
df_main = pd.DataFrame({'img_path':df_train_paths})
df_main['id'] = df_main['img_path'].apply(label_mapping)
# df_main['label'] = df_main['id'].apply(give_label)
# df_main = shuffle(df_main)
# df.head()
df = df.merge(df_main,on='id')
df.head()
df0 = df[df['label'] == 0].sample(50000,random_state=42)
df1 = df[df['label'] == 1].sample(50000,random_state=42)
df = pd.concat([df0,df1], ignore_index=True)
df.head()
df_train, df_val = train_test_split(df, random_state=42,test_size=0.5)
df_train.shape

df_val.shape
train_imgs = [cv2.imread(img_path) for img_path in list(df_train['img_path'])]
val_imgs = [cv2.imread(img_path) for img_path in list(df_val['img_path'])]
train_imgs = np.array(train_imgs)
val_imgs = np.array(val_imgs)
train_imgs.shape
val_imgs.shape
ia.seed(1)

# Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
# e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second
# image.
sometimes = lambda aug: iaa.Sometimes(0.5, aug)
def get_seq():
    # Define our sequence of augmentation steps that will be applied to every image.
    seq = iaa.Sequential([
        iaa.Fliplr(0.5), #horizontally flip 50% images
        iaa.Flipud(0.2), #vertically flip 20% images
        
        # crop some of the images by 0-10% of their height/width
        sometimes(iaa.Crop(percent=(0,0.1))),
        
        # Apply affine transformations to some of the images
        # - scale to 80-120% of image height/width (each axis independently)
        # - translate by -20 to +20 relative to height/width (per axis)
        # - rotate by -45 to +45 degrees
        # - shear by -16 to +16 degrees
        # - order: use nearest neighbour or bilinear interpolation (fast)
        # - mode: use any available mode to fill newly created pixels
        #         see API or scikit-image for which modes are available
        # - cval: if the mode is constant, then use a random brightness
        #         for the newly created pixels (e.g. sometimes black,
        #         sometimes white)
        sometimes(iaa.Affine(
            scale = {"x":(0.8,1.2), "y":(0.8,1.2)},
            translate_percent = {"x":(-0.2,0.2), "y":(-0.2,0.2)},
            rotate = (-45,45),
            shear = (-16,16),
            order = [0,1],
            cval = (0,255),
            mode = ia.ALL
        )),
        
        #
        # Execute 0 to 5 of the following (less important) augmenters per
        # image. Don't execute all of them, as that would often be way too
        # strong.
        #
        iaa.SomeOf((0,5),
                   [
                       # Convert some images into their superpixel representation,
                       # sample between 20 and 200 superpixels per image, but do
                       # not replace all superpixels with their average, only
                       # some of them (p_replace).
                       sometimes(
                           iaa.Superpixels(
                           p_replace = (0, 1.0),
                           n_segments = (20, 200)
                           )
                       ),
                       
                       # Blur each image with varying strength using
                       # gaussian blur (sigma between 0 and 3.0),
                       # average/uniform blur (kernel size between 2x2 and 7x7)
                       # median blur (kernel size between 3x3 and 11x11).
                       iaa.OneOf([
                           iaa.GaussianBlur((0,3.0)),
                           iaa.AverageBlur(k=(2,6)),
                           iaa.MedianBlur(k=(3,7))
                       ]),
                       
                       # Sharpen each image, overlay the result with the original
                       # image using an alpha between 0 (no sharpening) and 1
                       # (full sharpening effect).
                       iaa.Sharpen(alpha=(0,1.0), lightness = (0.75, 1.5)),
                       
                       # Same as sharpen, but for an embossing effect.
                       iaa.Emboss(alpha=(0,1.0), strength=(0,2.0)),
                       
                       # Search in some images either for all edges or for
                       # directed edges. These edges are then marked in a black
                       # and white image and overlayed with the original image
                       # using an alpha of 0 to 0.7.
                       sometimes(iaa.OneOf([
                           iaa.EdgeDetect(alpha = (0,0.7)),
                           iaa.DirectedEdgeDetect(alpha=(0,0.7), direction=(0.0,1.0))
                       ])),
                       
                       # Add gaussian noise to some images.
                       # In 50% of these cases, the noise is randomly sampled per
                       # channel and pixel.
                       # In the other 50% of all cases it is sampled once per
                       # pixel (i.e. brightness change).
                       iaa.AdditiveGaussianNoise(loc=0,scale=(0.0,0.05*255),per_channel=0.5),

                       # Either drop randomly 1 to 10% of all pixels (i.e. set
                       # them to black) or drop them on an image with 2-5% percent
                       # of the original size, leading to large dropped
                       # rectangles.
                       iaa.OneOf([
                           iaa.Dropout((0.01, 0.1),per_channel=0.5),
                           iaa.CoarseDropout((0.03,0.15),size_percent=(0.02,0.05), per_channel=0.2)
                       ]),
                       
                       # Invert each image's chanell with 5% probability.
                       # This sets each pixel value v to 255-v.
                       iaa.Invert(0.05, per_channel=True), #Invert colour channels
                       
                       # Add a value of -10 to 10 to each pixel.
                       iaa.Add((-10,10), per_channel=0.5),
                       
                       # Change brightness of images (50-150% of original value).
                       iaa.Multiply((0.5,1.5), per_channel=0.5),
                       
                       # Improve or worsen the contrast of images.
                       iaa.ContrastNormalization((0.5,2.0),per_channel=0.5),
                       
                       # Convert each image to grayscale and then overlay the
                       # result with the original with random alpha. I.e. remove
                       # colors with varying strengths.
                       iaa.Grayscale(alpha=(0.0,1.0)),
                       
                       # In some images move pixels locally around (with random
                       # strengths).
                       sometimes(iaa.ElasticTransformation(alpha=(0.5,3.5),sigma=0.25)),
                       
                       # In some images distort local areas with varying strength.
                       sometimes(iaa.PiecewiseAffine(scale=(0.01,0.05)))
                   ],
                   #do all the above augmentations in random order
                   random_order = True
                  )
            ],
            random_order = True
        )
    return seq
seq = get_seq()
train_imgs = seq.augment_images(train_imgs)

from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D, BatchNormalization
from keras.layers import Dropout, Flatten, Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential
from tensorflow import set_random_seed
set_random_seed(42)
model = Sequential()
model.add(Conv2D(filters = 16, kernel_size = 3, padding = 'same', activation = 'relu', input_shape = (96, 96, 3)))
model.add(Conv2D(filters = 16, kernel_size = 3, padding = 'same', activation = 'relu'))
model.add(Conv2D(filters = 16, kernel_size = 3, padding = 'same', activation = 'relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(MaxPooling2D(pool_size = 3)) 

model.add(Conv2D(filters = 32, kernel_size = 3, padding = 'same', activation = 'relu')) 
model.add(Conv2D(filters = 32, kernel_size = 3, padding = 'same', activation = 'relu')) 
model.add(Conv2D(filters = 32, kernel_size = 3, padding = 'same', activation = 'relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(MaxPooling2D(pool_size = 3)) 

model.add(Conv2D(filters = 64, kernel_size = 3, padding = 'same', activation = 'relu'))
model.add(Conv2D(filters = 64, kernel_size = 3, padding = 'same', activation = 'relu'))
model.add(Conv2D(filters = 64, kernel_size = 3, padding = 'same', activation = 'relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(MaxPooling2D(pool_size = 3))

model.add(Conv2D(filters = 128, kernel_size = 3, padding = 'same', activation = 'relu'))
model.add(Conv2D(filters = 128, kernel_size = 3, padding = 'same', activation = 'relu'))
model.add(Conv2D(filters = 256, kernel_size = 3, padding = 'same', activation = 'relu'))
# model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(MaxPooling2D(pool_size = 3))
model.add(Flatten())
model.add(Dense(1, activation = 'sigmoid'))
# define roc_callback, inspired by https://github.com/keras-team/keras/issues/6050#issuecomment-329996505
def auc_roc(y_true, y_pred):
    # any tensorflow metric
    value, update_op = tf.contrib.metrics.streaming_auc(y_pred, y_true)

    # find all variables created for this metric
    metric_vars = [i for i in tf.local_variables() if 'auc_roc' in i.name.split('/')[1]]

    # Add metric variables to GLOBAL_VARIABLES collection.
    # They will be initialized for new session.
    for v in metric_vars:
        tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)

    # force to update metric values
    with tf.control_dependencies([update_op]):
        value = tf.identity(value)
        return value
my_callbacks = [EarlyStopping(monitor='auc_roc', patience=300, verbose=1, mode='max')]

model.compile(optimizer= 'adam', loss='binary_crossentropy', metrics=['accuracy',auc_roc])

batch_size = 128
model.fit(train_imgs,df_train['label'], batch_size = batch_size,epochs=28,validation_data=(val_imgs,df_val['label']), callbacks=my_callbacks)
pred = (model.predict(val_imgs).ravel()*model.predict(val_imgs[:,::-1,:,:]).ravel()*model.predict(val_imgs[:,:,::-1,:]).ravel()*model.predict(val_imgs[:,::-1,::-1,:]).ravel())**0.25

roc_auc_score(df_val['label'],pred)

test_imgs = [cv2.imread(img_path) for img_path in df_test_paths]
test_imgs = np.array(test_imgs)
test_imgs.shape
predtest = (model.predict(test_imgs).ravel()*model.predict(test_imgs[:,::-1,:,:]).ravel()*model.predict(test_imgs[:,:,::-1,:]).ravel()*model.predict(test_imgs[:,::-1,::-1,:]).ravel())**0.25
id = []
for path in df_test_paths:
    id.append(label_mapping(path))


submit = pd.DataFrame({'id':id,'label':predtest})
submit.head()
submit.to_csv("sub_new.csv",index=False)
