# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/carvana-image-masking-challenge/train/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!unzip /kaggle/input/carvana-image-masking-challenge/train.zip -d /kaggle/carvana-image-masking-challenge/
!unzip /kaggle/input/carvana-image-masking-challenge/train_masks.zip -d /kaggle/carvana-image-masking-challenge/
import glob
import pandas as pd
import os
import pandas as pd
from sklearn.utils import shuffle
import numpy as np
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import concatenate, Conv2DTranspose, Input, Dense, Dropout, Flatten, Activation, Conv2D, MaxPooling2D, GlobalAveragePooling2D, BatchNormalization,GlobalMaxPooling2D
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras import utils
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg16 import VGG16
import matplotlib.pyplot as plt
%matplotlib inline
from keras import backend as K
files_path = '../carvana-image-masking-challenge/train'
target_files_path = '../carvana-image-masking-challenge/train_masks'
data_files = {}
data_target = {}
data_files['files_path'] = []
data_target['target_files_path'] = []
data_files['files_path'] = list(glob.glob(files_path + "/*"))
data_target['target_files_path'] = list(glob.glob(target_files_path + "/*"))

data_files = pd.DataFrame(data_files)
data_target = pd.DataFrame(data_target)
def file_name(x):
    return x.split("/")[-1].split(".")[0]
data_files["file_name"] = data_files["files_path"].apply(lambda x: file_name(x))
data_target["file_name"] = data_target["target_files_path"].apply(lambda x: file_name(x)[:-5])
data = pd.merge(data_files, data_target, on = "file_name", how = "inner")
data.head()
#data = data.sample(frac=0.3, replace=False, random_state=42)
n = int(round(data.shape[0] * 0.7,0))
data_train = data[0:n]
data_test = data[n:]
images_test = np.array([img_to_array(
                    load_img(img, target_size=(256,256))
                    ) for img in data_test['files_path'].values.tolist()])


images_train = np.array([img_to_array(
                    load_img(img, target_size=(256,256))
                    ) for img in data_train['files_path'].values.tolist()])
images_train = images_train.astype('float32')/255.0
images_test = images_test.astype('float32')/255.0

images_test_target = np.array([np.average(img_to_array(
                    load_img(img, target_size=(256,256))
                    )/255, axis=-1) for img in data_test['target_files_path'].values.tolist()])

#images_test_target = images_test_target.astype('bool')/255
#images_test_target = np.average(images_test_target, axis=-1)
images_train_target = np.array([np.average(img_to_array(
                    load_img(img, target_size=(256,256))
                    )/255, axis=-1) for img in data_train['target_files_path'].values.tolist()])
images_train_target = images_train_target[:,:,:,None]
images_test_target = images_test_target[:,:,:,None]
images_test_target[0].shape
#images_train_target_a = None
#images_test_target_a = None

import gc
gc.collect()
fig, axes = plt.subplots(ncols=2, figsize=(12, 12))
ax1, ax2 = axes
ax1.imshow(images_train[0]);
#ax1.set_grid(True);
ax1.set_xticks([]);
ax1.set_yticks([]);
ax1.set_title("Original Image Train")

ax2.imshow(np.squeeze(images_train_target[0]));
#ax2.set_grid(True);
ax2.set_xticks([]);
ax2.set_yticks([])
ax2.set_title("Mask")

fig, axes = plt.subplots(ncols=2, figsize=(12, 12))
ax1, ax2 = axes
ax1.imshow(images_test[0]);
#ax1.set_grid(True);
ax1.set_xticks([]);
ax1.set_yticks([]);
ax1.set_title("Original Image Test")

ax2.imshow(np.squeeze(images_test_target[0]));
#ax2.set_grid(True);
ax2.set_xticks([]);
ax2.set_yticks([])
ax2.set_title("Mask")

IMG_HEIGHT = 256
IMG_WIDTH = 256
IMG_CHANNELS = 3
def iou_loss_score(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    union = K.sum(y_true,-1) + K.sum(y_pred,-1) - intersection
    iou = (intersection + smooth) / ( union + smooth)
    return iou
inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
#s = Lambda(lambda x: x / 255) (inputs)

c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (inputs)
c1 = Dropout(0.1) (c1)
c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c1)
p1 = MaxPooling2D((2, 2)) (c1)

c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p1)
c2 = Dropout(0.1) (c2)
c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c2)
p2 = MaxPooling2D((2, 2)) (c2)

c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p2)
c3 = Dropout(0.2) (c3)
c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c3)
p3 = MaxPooling2D((2, 2)) (c3)

c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p3)
c4 = Dropout(0.2) (c4)
c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c4)
p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p4)
c5 = Dropout(0.3) (c5)
c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c5)

u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c5)
u6 = concatenate([u6, c4])
c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u6)
c6 = Dropout(0.2) (c6)
c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c6)

u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c6)
u7 = concatenate([u7, c3])
c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u7)
c7 = Dropout(0.2) (c7)
c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c7)

u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c7)
u8 = concatenate([u8, c2])
c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u8)
c8 = Dropout(0.1) (c8)
c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c8)

u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c8)
u9 = concatenate([u9, c1], axis=3)
c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u9)
c9 = Dropout(0.1) (c9)
c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c9)

outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)

model = Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[iou_loss_score])
model.summary()
model.fit(images_train, images_train_target, epochs = 10, batch_size = 64, validation_data = (images_test, images_test_target))
preds_train = (model.predict(images_train[0][None]) > 0.5).astype(np.uint8)
fig, axes = plt.subplots(ncols=3, figsize=(12, 12))
ax1, ax2, ax3 = axes
ax1.imshow(images_train[0]);
#ax1.set_grid(True);
ax1.set_xticks([]);
ax1.set_yticks([]);
ax1.set_title("Original Image Train")

ax2.imshow(np.squeeze(images_train_target[0]));
#ax2.set_grid(True);
ax2.set_xticks([]);
ax2.set_yticks([])
ax2.set_title("Mask")


ax3.imshow(np.squeeze(preds_train[0]));
#ax2.set_grid(True);
ax3.set_xticks([]);
ax3.set_yticks([])
ax3.set_title("Predicted Mask")


preds_test = (model.predict(images_test[0][None]) > 0.5).astype(np.uint8)
fig, axes = plt.subplots(ncols=3, figsize=(12, 12))
ax1, ax2, ax3 = axes
ax1.imshow(images_test[0]);
#ax1.set_grid(True);
ax1.set_xticks([]);
ax1.set_yticks([]);
ax1.set_title("Original Image Test")

ax2.imshow(np.squeeze(images_test_target[0]));
#ax2.set_grid(True);
ax2.set_xticks([]);
ax2.set_yticks([])
ax2.set_title("Mask")


ax3.imshow(np.squeeze(preds_test[0]));
#ax2.set_grid(True);
ax3.set_xticks([]);
ax3.set_yticks([])
ax3.set_title("Predicted Mask")

