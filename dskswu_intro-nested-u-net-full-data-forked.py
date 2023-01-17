# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

import glob

import keras

import pydicom



print(os.listdir("../input/siim-acr-pneumothorax-segmentation"))

print()

print(os.listdir("../input/siim-acr-pneumothorax-segmentation/sample images"))

# Any results you write to the current directory are saved as output.



from matplotlib import cm

from matplotlib import pyplot as plt



from keras.models import Model

from keras.layers import Input

from keras.layers.convolutional import Conv2D, Conv2DTranspose

from keras.layers.pooling import MaxPooling2D

from keras.layers.merge import concatenate

from keras import backend as K



import keras

import tensorflow as tf

from keras.models import Model

from keras import backend as K

from keras.layers import Input, merge, Conv2D, ZeroPadding2D, UpSampling2D, Dense, concatenate, Conv2DTranspose

from keras.layers.pooling import MaxPooling2D, GlobalAveragePooling2D, MaxPooling2D

from keras.layers.core import Dense, Dropout, Activation

from keras.layers import BatchNormalization, Dropout, Flatten, Lambda

from keras.layers.advanced_activations import ELU, LeakyReLU

from keras.optimizers import Adam, RMSprop, SGD

from keras.regularizers import l2

from keras.layers.noise import GaussianDropout



import numpy as np



smooth = 1.

dropout_rate = 0.3

act = "relu"





import tensorflow as tf



from tqdm import tqdm_notebook



import sys

sys.path.insert(0, '../input/siim-acr-pneumothorax-segmentation')



from mask_functions import rle2mask
def show_dcm_info(dataset):

    print("Filename.........:", file_path)

    print("Storage type.....:", dataset.SOPClassUID)

    print()



    pat_name = dataset.PatientName

    display_name = pat_name.family_name + ", " + pat_name.given_name

    print("Patient's name......:", display_name)

    print("Patient id..........:", dataset.PatientID)

    print("Patient's Age.......:", dataset.PatientAge)

    print("Patient's Sex.......:", dataset.PatientSex)

    print("Modality............:", dataset.Modality)

    print("Body Part Examined..:", dataset.BodyPartExamined)

    print("View Position.......:", dataset.ViewPosition)

    

    if 'PixelData' in dataset:

        rows = int(dataset.Rows)

        cols = int(dataset.Columns)

        print("Image size.......: {rows:d} x {cols:d}, {size:d} bytes".format(

            rows=rows, cols=cols, size=len(dataset.PixelData)))

        if 'PixelSpacing' in dataset:

            print("Pixel spacing....:", dataset.PixelSpacing)



def plot_pixel_array(dataset, figsize=(10,10)):

    plt.figure(figsize=figsize)

    plt.imshow(dataset.pixel_array, cmap=plt.cm.bone)

    plt.show()

for file_path in glob.glob('../input/siim-acr-pneumothorax-segmentation/sample images/*.dcm'):

    dataset = pydicom.dcmread(file_path)

    show_dcm_info(dataset)

    plot_pixel_array(dataset)

    break # Comment this out to see all
num_img = len(glob.glob('../input/siim-acr-pneumothorax-segmentation/sample images/*.dcm'))

fig, ax = plt.subplots(nrows=1, ncols=num_img, sharey=True, figsize=(num_img*10,10))

for q, file_path in enumerate(glob.glob('../input/siim-acr-pneumothorax-segmentation/sample images/*.dcm')):

    dataset = pydicom.dcmread(file_path)

    #show_dcm_info(dataset)

    

    ax[q].imshow(dataset.pixel_array, cmap=plt.cm.bone)
start = 4   # Starting index of images

num_img = 5 # Total number of images to show



fig, ax = plt.subplots(nrows=1, ncols=num_img, sharey=True, figsize=(num_img*10,10))

for q, file_path in enumerate(glob.glob('../input/siim-acr-pneumothorax-segmentation/sample images/*.dcm')[start:start+num_img]):

    dataset = pydicom.dcmread(file_path)

    #show_dcm_info(dataset)

    

    ax[q].imshow(dataset.pixel_array, cmap=plt.cm.bone)

df = pd.read_csv('../input/siim-acr-pneumothorax-segmentation/sample images/train-rle-sample.csv', header=None, index_col=0)



fig, ax = plt.subplots(nrows=1, ncols=num_img, sharey=True, figsize=(num_img*10,10))

for q, file_path in enumerate(glob.glob('../input/siim-acr-pneumothorax-segmentation/sample images/*.dcm')[start:start+num_img]):

    dataset = pydicom.dcmread(file_path)

    #print(file_path.split('/')[-1][:-4])

    ax[q].imshow(dataset.pixel_array, cmap=plt.cm.bone)

    if df.loc[file_path.split('/')[-1][:-4],1] != '-1':

        mask = rle2mask(df.loc[file_path.split('/')[-1][:-4],1], 1024, 1024).T

        ax[q].set_title('See Marker')

        ax[q].imshow(mask, alpha=0.1, cmap="Reds")

    else:

        ax[q].set_title('Nothing to see')

train_glob = '../input/siim-acr-pneumothorax-segmentation-data/pneumothorax/dicom-images-train/*/*/*.dcm'

test_glob = '../input/siim-acr-pneumothorax-segmentation-data/pneumothorax/dicom-images-test/*/*/*.dcm'

train_fns = sorted(glob.glob(train_glob))[:5000]

test_fns = sorted(glob.glob(test_glob))[:5000]

df_full = pd.read_csv('../input/siim-acr-pneumothorax-segmentation-data/pneumothorax/train-rle.csv', index_col='ImageId')
df_full.columns
im_height = 1024

im_width = 1024

im_chan = 1

# Get train images and masks

X_train = np.zeros((len(train_fns), im_height, im_width, im_chan), dtype=np.uint8)

Y_train = np.zeros((len(train_fns), im_height, im_width, 1), dtype=np.bool)

print('Getting train images and masks ... ')

sys.stdout.flush()

for n, _id in tqdm_notebook(enumerate(train_fns), total=len(train_fns)):

    dataset = pydicom.read_file(_id)

    X_train[n] = np.expand_dims(dataset.pixel_array, axis=2)

    try:

        if '-1' in df_full.loc[_id.split('/')[-1][:-4],' EncodedPixels']:

            Y_train[n] = np.zeros((1024, 1024, 1))

        else:

            if type(df_full.loc[_id.split('/')[-1][:-4],' EncodedPixels']) == str:

                Y_train[n] = np.expand_dims(rle2mask(df_full.loc[_id.split('/')[-1][:-4],' EncodedPixels'], 1024, 1024), axis=2)

            else:

                Y_train[n] = np.zeros((1024, 1024, 1))

                for x in df_full.loc[_id.split('/')[-1][:-4],' EncodedPixels']:

                    Y_train[n] =  Y_train[n] + np.expand_dims(rle2mask(x, 1024, 1024), axis=2)

    except KeyError:

        print(f"Key {_id.split('/')[-1][:-4]} without mask, assuming healthy patient.")

        Y_train[n] = np.zeros((1024, 1024, 1)) # Assume missing masks are empty masks.



print('Done!')
im_height = 256

im_width = 256

X_train = X_train.reshape((-1, im_height, im_width, 1))

Y_train = Y_train.reshape((-1, im_height, im_width, 1))
def dice_coef(y_true, y_pred, smooth=1):

    y_true_f = K.flatten(y_true)

    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f)

    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)





def dice_loss(y_true, y_pred):

    smooth = 1.

    y_true_f = K.flatten(y_true)

    y_pred_f = K.flatten(y_pred)

    intersection = y_true_f * y_pred_f

    score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

    return 1. - score



def bce_dice_loss(y_true, y_pred):

    return keras.losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)







#bce



def bce(y_pred):

    #see https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/keras/backend.py#L3525

    y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())

    return tf.log(y_pred / (1 - y_pred))



def bceloss(y_true, y_pred):

    beta = 2.0

    y_pred = bce(y_pred)

    pos_weight = beta / (1 - beta)

    loss = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred, targets=y_true, pos_weight=pos_weight)



    return tf.reduce_mean(loss * (1 - beta))



#https://github.com/MrGiovanni/UNetPlusPlus.git

# 2D Standard

def standard_unit(input_tensor, stage, nb_filter, kernel_size=3):



    x = Conv2D(nb_filter, (kernel_size, kernel_size), activation=act, name='conv'+stage+'_1', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(input_tensor)

    x = Dropout(dropout_rate, name='dp'+stage+'_1')(x)

    x = Conv2D(nb_filter, (kernel_size, kernel_size), activation=act, name='conv'+stage+'_2', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(x)

    x = Dropout(dropout_rate, name='dp'+stage+'_2')(x)



    return x
"""

Standard UNet++ [Zhou et.al, 2018]

Total params: 9,041,601

"""

def Nest_Net(img_rows, img_cols, color_type=1, num_class=1, deep_supervision=False):



    nb_filter = [16,32,64,128,256,512]



    # Handle Dimension Ordering for different backends

    global bn_axis

    if K.image_dim_ordering() == 'tf':

      bn_axis = 3

      img_input = Input(shape=(img_rows, img_cols, color_type), name='main_input')

    else:

      bn_axis = 1

      img_input = Input(shape=(color_type, img_rows, img_cols), name='main_input')



    conv1_1 = standard_unit(img_input, stage='11', nb_filter=nb_filter[0])

    pool1 = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(conv1_1)



    conv2_1 = standard_unit(pool1, stage='21', nb_filter=nb_filter[1])

    pool2 = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(conv2_1)



    up1_2 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up12', padding='same')(conv2_1)

    conv1_2 = concatenate([up1_2, conv1_1], name='merge12', axis=bn_axis)

    conv1_2 = standard_unit(conv1_2, stage='12', nb_filter=nb_filter[0])



    conv3_1 = standard_unit(pool2, stage='31', nb_filter=nb_filter[2])

    pool3 = MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(conv3_1)



    up2_2 = Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up22', padding='same')(conv3_1)

    conv2_2 = concatenate([up2_2, conv2_1], name='merge22', axis=bn_axis)

    conv2_2 = standard_unit(conv2_2, stage='22', nb_filter=nb_filter[1])



    up1_3 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up13', padding='same')(conv2_2)

    conv1_3 = concatenate([up1_3, conv1_1, conv1_2], name='merge13', axis=bn_axis)

    conv1_3 = standard_unit(conv1_3, stage='13', nb_filter=nb_filter[0])



    conv4_1 = standard_unit(pool3, stage='41', nb_filter=nb_filter[3])

    pool4 = MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(conv4_1)



    up3_2 = Conv2DTranspose(nb_filter[2], (2, 2), strides=(2, 2), name='up32', padding='same')(conv4_1)

    conv3_2 = concatenate([up3_2, conv3_1], name='merge32', axis=bn_axis)

    conv3_2 = standard_unit(conv3_2, stage='32', nb_filter=nb_filter[2])



    up2_3 = Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up23', padding='same')(conv3_2)

    conv2_3 = concatenate([up2_3, conv2_1, conv2_2], name='merge23', axis=bn_axis)

    conv2_3 = standard_unit(conv2_3, stage='23', nb_filter=nb_filter[1])



    up1_4 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up14', padding='same')(conv2_3)

    conv1_4 = concatenate([up1_4, conv1_1, conv1_2, conv1_3], name='merge14', axis=bn_axis)

    conv1_4 = standard_unit(conv1_4, stage='14', nb_filter=nb_filter[0])



    conv5_1 = standard_unit(pool4, stage='51', nb_filter=nb_filter[4])



    up4_2 = Conv2DTranspose(nb_filter[3], (2, 2), strides=(2, 2), name='up42', padding='same')(conv5_1)

    conv4_2 = concatenate([up4_2, conv4_1], name='merge42', axis=bn_axis)

    conv4_2 = standard_unit(conv4_2, stage='42', nb_filter=nb_filter[3])



    up3_3 = Conv2DTranspose(nb_filter[2], (2, 2), strides=(2, 2), name='up33', padding='same')(conv4_2)

    conv3_3 = concatenate([up3_3, conv3_1, conv3_2], name='merge33', axis=bn_axis)

    conv3_3 = standard_unit(conv3_3, stage='33', nb_filter=nb_filter[2])



    up2_4 = Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up24', padding='same')(conv3_3)

    conv2_4 = concatenate([up2_4, conv2_1, conv2_2, conv2_3], name='merge24', axis=bn_axis)

    conv2_4 = standard_unit(conv2_4, stage='24', nb_filter=nb_filter[1])



    up1_5 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up15', padding='same')(conv2_4)

    conv1_5 = concatenate([up1_5, conv1_1, conv1_2, conv1_3, conv1_4], name='merge15', axis=bn_axis)

    conv1_5 = standard_unit(conv1_5, stage='15', nb_filter=nb_filter[0])



    nestnet_output_1 = Conv2D(num_class, (1, 1), activation='sigmoid', name='output_1', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(conv1_2)

    nestnet_output_2 = Conv2D(num_class, (1, 1), activation='sigmoid', name='output_2', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(conv1_3)

    nestnet_output_3 = Conv2D(num_class, (1, 1), activation='sigmoid', name='output_3', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(conv1_4)

    nestnet_output_4 = Conv2D(num_class, (1, 1), activation='sigmoid', name='output_4', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(conv1_5)



    if deep_supervision:

        model = Model(input=img_input, output=[nestnet_output_1,

                                               nestnet_output_2,

                                               nestnet_output_3,

                                               nestnet_output_4])

    else:

        model = Model(input=img_input, output=[nestnet_output_4])



    return model
model = Nest_Net(None, None, im_chan)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[bceloss])

#model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[balanced_cross_entropy])

model.summary()
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau

from sklearn.model_selection import train_test_split





checkpoint = ModelCheckpoint('../working/unet.h5', monitor='val_loss', verbose=1, 

                             save_best_only=True, mode='min', save_weights_only = True)

reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, 

                                   verbose=1, mode='auto', epsilon=0.0001)

early = EarlyStopping(monitor="val_loss", 

                      mode="min", 

                      patience=6)

callbacks_list = [checkpoint, early, reduceLROnPlat]
model.fit (X_train, Y_train, validation_split=.2, batch_size=32, epochs=4)
import cv2

from PIL import *

import PIL

from mask_functions import rle2mask,mask2rle



img_size = 512

def test_images_pred(test_fns):

    pred_rle = []

    ids = []

    model.load_weights('../working/unet.h5')

    for f in tqdm_notebook(test_fns):

        img = pydicom.read_file(f).pixel_array

        img = cv2.resize(img,(img_size,img_size))

        img = model.predict(img.reshape(1,img_size,img_size,1))

        img = img.reshape(img_size,img_size)

        ids.append('.'.join(f.split('/')[-1].split('.')[:-1]))

        #img = PIL.Image.fromarray(((img.T*255).astype(np.uint8)).resize(1024,1024))

        img = PIL.Image.fromarray((img.T*255).astype(np.uint8)).resize((1024,1024))

        img = np.asarray(img)

        pred_rle.append(mask2rle(img,1024,1024))

    return pred_rle,ids
preds,ids = test_images_pred(test_fns)
print(preds[10])

print(len(preds),len(ids))
submission1 = pd.DataFrame({'ImageId':ids,'EncodedPixels':preds})
submission1.head()
submission1.to_csv('newsubmit.csv',index = False)
from IPython.display import HTML

html = "<a href = submission.csv>d</a>"

HTML(html)
df = submission1

# import the modules we'll need

from IPython.display import HTML

import pandas as pd

import numpy as np

import base64



# function that takes in a dataframe and creates a text link to  

# download it (will only work for files < 2MB or so)

def create_download_link(df, title = "Download CSV file", filename = "mobassir_submission.csv"):  

    csv = df.to_csv()

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)



# create a random sample dataframe

df = pd.DataFrame(sub_df1)



# create a link to download the dataframe

create_download_link(df)