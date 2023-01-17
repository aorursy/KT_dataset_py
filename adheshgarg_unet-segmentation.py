# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/bsd300t/bsd300t/BSDS300/images/train"))



# Any results you write to the current directory are saved as output.
import os

import sys

import random

import warnings



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline



import cv2



from tqdm import tqdm_notebook, tnrange

from itertools import chain

from skimage.io import imread, imshow, concatenate_images

from skimage.transform import resize

from skimage.morphology import label

from sklearn.model_selection import train_test_split



from keras.models import Model, load_model

from keras.layers import Input

from keras.layers.core import Lambda, RepeatVector, Reshape

from keras.layers.convolutional import Conv2D, Conv2DTranspose

from keras.layers.pooling import MaxPooling2D

from keras.layers.merge import concatenate

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from keras import backend as K



import tensorflow as tf



from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img



%matplotlib inline

import numpy as np

import matplotlib.pyplot as plt

import scipy.misc

from zipfile import ZipFile

from io import BytesIO



# Image manipulation.

import PIL.Image

from IPython.display import display
def reshape_image(image_file, new_wigth, new_height):

    

    olddim = np.shape(image_file)

    img = np.zeros((new_wigth,new_height))

    newdim = np.shape(img)

        

    for r in range(newdim[0]):

        if (newdim[0] <= olddim[0]):

            centerx=(r)/newdim[0]*olddim[0]

            lowerx=max(0,int(round(centerx-olddim[0]/newdim[0]/2,0)))

            upperx=min(olddim[0],int(round(centerx+olddim[0]/newdim[0]/2,0))+1)

        else:

            lowerx=max(0,int(r*olddim[0]/newdim[0]))

            upperx=min(lowerx+1,olddim[0]-1)+1

            

        for c in range(newdim[1]):  

            if (newdim[1] <= olddim[1]):

                centery=(c)/newdim[1]*olddim[1]

                lowery=max(0,int(round(centery-olddim[1]/newdim[1]/2,0)))

                uppery=min(olddim[1],int(round(centery+olddim[1]/newdim[1]/2,0))+1)

            else:

                lowery=max(0,int(c*olddim[1]/newdim[1]))

                uppery=min(lowery+1,olddim[1]-1)+1

            img[r,c] = np.mean(image_file[ lowerx:upperx, lowery:uppery ])



                

    return img
df_depths = pd.read_csv('../input/bsd300t/bsd300t/BSDS300/iids_test.csv')

df_depths.head()
ids= ['170057','58060','163085',]

plt.figure(figsize=(30,15))

for j, img_name in enumerate(ids):

    q = j+1

    img = load_img('../input/bsd300t/bsd300t/BSDS300/images/test/' + img_name + '.jpg')

    

    img = np.array(img)

    img_cumsum = (np.float32(img)-img.mean()).cumsum(axis=0)

    

    plt.subplot(1,3*(1+len(ids)),q*3-2)

    plt.imshow(img, cmap='seismic')

    plt.subplot(1,3*(1+len(ids)),q*3-1)

    plt.imshow(img_cumsum, cmap='seismic')

plt.show()
im_width = 481

im_height = 481

#border = 5

im_chan = 2 # Number of channels: first is original and second cumsum(axis=0)

path_train = '../input/bsd300t/bsd300t/BSDS300/images/train/'

path_test = '../input/bsd300t/bsd300t/BSDS300/images/test/'
train_ids = next(os.walk(path_train))[2]

test_ids = next(os.walk(path_test))[2]
X = np.zeros((len(train_ids), im_height, im_width, im_chan), dtype=np.float32)

print('Getting and resizing train images and masks ... ')

sys.stdout.flush()

for n, id_ in tqdm_notebook(enumerate(train_ids), total=len(train_ids)):

    path = path_train

    

    # Load X

    img = load_img(path + id_, grayscale=True)

    x_img = img_to_array(img)

    x_img = reshape_image(x_img, 481, 481)

    # Save images

    X[n, ..., 0] = x_img.squeeze() / 255



print('Done!')
X_train = train_test_split(X, test_size=0.15, random_state=42)
ix = random.randint(0, len(X_train))



has_mask = y_train[ix].max() > 0



fig, ax = plt.subplots(1, 3, figsize=(20, 10))

ax[0].imshow(X_train[ix, ..., 0], cmap='seismic', interpolation='bilinear')

if has_mask:

    ax[0].contour(y_train[ix].squeeze(), colors='k', levels=[0.5])

ax[0].set_title('Seismic')



ax[1].imshow(X_train[ix, ..., 1], cmap='seismic', interpolation='bilinear')

if has_mask:

    ax[1].contour(y_train[ix].squeeze(), colors='k', levels=[0.5])

ax[1].set_title('Seismic cumsum')



ax[2].imshow(y_train[ix].squeeze(), interpolation='bilinear', cmap='gray')

ax[2].set_title('Salt');
def mean_iou(y_true, y_pred):

    prec = []

    for t in np.arange(0.5, 1.0, 0.05):

        y_pred_ = tf.to_int32(y_pred > t)

        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)

        K.get_session().run(tf.local_variables_initializer())

        with tf.control_dependencies([up_opt]):

            score = tf.identity(score)

        prec.append(score)

    return K.mean(K.stack(prec), axis=0)

input_img = Input((im_height, im_width, im_chan), name='img')

input_features = Input((n_features, ), name='feat')



c1 = Conv2D(8, (3, 3), activation='relu', padding='same') (input_img)

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



# Join features information in the depthest layer

f_repeat = RepeatVector(8*8)(input_features)

f_conv = Reshape((8, 8, n_features))(f_repeat)

p4_feat = concatenate([p4, f_conv], -1)



c5 = Conv2D(128, (3, 3), activation='relu', padding='same') (p4_feat)

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



model = Model(inputs=[input_img, input_features], outputs=[outputs])

model.compile(optimizer='adam', loss='binary_crossentropy') #, metrics=[mean_iou]) # The mean_iou metrics seens to leak train and test values...

model.summary()
callbacks = [

    EarlyStopping(patience=5, verbose=1),

    ReduceLROnPlateau(patience=3, verbose=1),

    ModelCheckpoint('model-tgs-salt-1.h5', verbose=1, save_best_only=True, save_weights_only=True)

]



results = model.fit({'img': X_train, 'feat': X_feat_train}, y_train, batch_size=16, epochs=50, callbacks=callbacks,

                    validation_data=({'img': X_valid, 'feat': X_feat_valid}, y_valid))