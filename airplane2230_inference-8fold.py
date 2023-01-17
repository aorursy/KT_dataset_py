import gc

import os

import warnings

import numpy as np 

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



import math

from keras.callbacks import Callback

from keras import backend

from keras.models import Sequential, Model

from keras.layers import Dense, Dropout, Flatten, Activation, Conv2D, GlobalAveragePooling2D, BatchNormalization, Input

from keras.optimizers import Adam, SGD, Nadam

from keras.metrics import categorical_accuracy

from keras.callbacks import ModelCheckpoint, EarlyStopping

from keras.callbacks import LearningRateScheduler, ReduceLROnPlateau, TensorBoard



import cv2



import PIL

from PIL import ImageOps, ImageFilter, ImageDraw

from keras.applications import *

from sklearn.model_selection import train_test_split

from keras.preprocessing.image import ImageDataGenerator

from keras import backend as K

warnings.filterwarnings(action='ignore')



!pip install git+https://github.com/qubvel/efficientnet

from efficientnet.keras import EfficientNetB3



K.image_data_format()
list_dir = os.listdir('../input')

print(list_dir)
CROP_PATH = os.path.join('../input', list_dir[2])

CSV_PATH = os.path.join('../input', list_dir[6])



# image folder path

TRAIN_IMG_PATH = CROP_PATH + '/train_crop'

TEST_IMG_PATH = CROP_PATH + '/test_crop'
model_path = './model/'



if(not os.path.exists(model_path)):

    os.mkdir(model_path)



# read csv

df_train = pd.read_csv(CSV_PATH + '/train.csv')

df_test = pd.read_csv(CSV_PATH + '/test.csv')

df_class = pd.read_csv(CSV_PATH + '/class.csv')



df_train["class"] = df_train["class"].astype('str')



df_train = df_train[['img_file', 'class']]

df_test = df_test[['img_file']]



# Parameter

# nb_train_samples = len(X_train)

# nb_validation_samples = len(X_val)

nb_test_samples = len(df_test)



# Define Generator config

# https://www.kaggle.com/kozistr/seedlings-densenet-161-48-public-lb-98-236



batch_size = 32



train_datagen = ImageDataGenerator(

    rotation_range = 60,

#     shear_range = 0.25,

    width_shift_range=0.30,

    height_shift_range=0.30,

    horizontal_flip = True, 

    vertical_flip = False,

    zoom_range=0.25,

    fill_mode = 'nearest',

    rescale = 1./255)



val_datagen = ImageDataGenerator(rescale = 1./255)

test_datagen = ImageDataGenerator(rescale = 1./255)



def get_steps(num_samples, batch_size):

    if (num_samples % batch_size) > 0 :

        return (num_samples // batch_size) + 1

    else :

        return num_samples // batch_size
img_size = (299, 299)

# Make Generator

train_generator_299 = train_datagen.flow_from_dataframe(

    dataframe=df_train, 

    directory=TRAIN_IMG_PATH,

    x_col = 'img_file',

    y_col = 'class',

    target_size = img_size,

    color_mode='rgb',

    class_mode='categorical',

    batch_size=batch_size,

    seed=42

)

    

test_generator_299 = test_datagen.flow_from_dataframe(

    dataframe=df_test,

    directory=TEST_IMG_PATH,

    x_col='img_file',

    y_col=None,

    target_size= (299, 299),

    color_mode='rgb',

    class_mode=None,

    batch_size=batch_size,

    shuffle=False

)



test_generator_224 = test_datagen.flow_from_dataframe(

    dataframe=df_test,

    directory=TEST_IMG_PATH,

    x_col='img_file',

    y_col=None,

    target_size= (224, 224),

    color_mode='rgb',

    class_mode=None,

    batch_size=batch_size,

    shuffle=False

)
def get_model(base_model, input_size, train_session, dropout_rate = 0.3):

    base_model = base_model(weights='imagenet', input_shape=(input_size,input_size,3), include_top=False)



    inputs = Input(shape = (input_size, input_size, 3), name = 'input_1')

    x = base_model(inputs)

    x = GlobalAveragePooling2D()(x)

    x = Dense(2048, kernel_initializer='he_normal')(x)

    x = Dropout(dropout_rate)(x)

    x = Activation('relu')(x)

    x = Dense(196, activation = 'softmax')(x)



    model = Model(inputs = inputs, outputs = x)

    if(train_session):

        nadam = Nadam(lr = lr)

        model.compile(optimizer= nadam, loss='categorical_crossentropy', metrics=[categorical_accuracy,

                                                                              f1_m, precision_m, recall_m])

    return model
import os

os.listdir('../input')
from keras.models import load_model



# MODEL folder path

EFF_PATH = os.path.join('../input', list_dir[5])

XCEPTION_PATH = os.path.join('../input', list_dir[1])

INCEPRES_PATH = os.path.join('../input', list_dir[4])

CUTMIX_PATH = os.path.join('../input', list_dir[1])



xception_model = ['_Xception_f1_8fold.hdf5', Xception]

eff_model = ['_EFF_f1_8fold.hdf5', EfficientNetB3]

incepres_model = ['_IncepRes_f1_8fold.hdf5', InceptionResNetV2]

cutmix_model = ['_cmm.hdf5', InceptionResNetV2]



model_list = [xception_model, eff_model, incepres_model]

# total predictions list

preds_list = []



TTA_STEPS = 5



for model_name, base_model in model_list:

    print(model_name)

    # prediction each fold

    img_size = 299

    predictions = []

    fold_num = 8 + 1

    

    # model_load_dir

    if(model_name == '_EFF_f1_8fold.hdf5'):

        model_load_dir = EFF_PATH

    elif(model_name == '_Xception_f1_8fold.hdf5'):

        model_load_dir = XCEPTION_PATH

    elif(model_name == '_IncepRes_f1_8fold.hdf5'):

        model_load_dir = INCEPRES_PATH

    elif(model_name == '_cmm.hdf5'):

        model_load_dir = CUTMIX_PATH



    for i in range(1, fold_num):

        model = get_model(base_model, img_size, train_session = False)

        # '..input/EFF_F1_8fold/i_EFF_f1_8fold.hdf5'

        model.load_weights(os.path.join(model_load_dir, str(i)) + model_name)

        # tta prediction list

        tta_preds = []

        for _ in range(TTA_STEPS):

            if(img_size == 224):

                test_generator_224.reset()

                pred = model.predict_generator(

                generator = test_generator_224, 

                steps = get_steps(nb_test_samples, batch_size),

                verbose = 1

                )

            else:

                test_generator_299.reset()

                pred = model.predict_generator(

                generator = test_generator_299, 

                steps = get_steps(nb_test_samples, batch_size),

                verbose = 1

                )

            tta_preds.append(pred) # (TTA_STEP, 6150, 196)

        tta_preds = np.mean(tta_preds, axis = 0) # (6150, 196)

        predictions.append(tta_preds) # (fold, 6150, 196)

        # for memory leaky

        del model

        for _ in range(10):

            gc.collect()

        K.clear_session()

    preds_list.append(np.mean(predictions, axis = 0))
# 0.34, 0.34, 0.32가 제일 좋음.

preds = (preds_list[1] * 0.5) + (preds_list[2] * 0.5)

preds_class_indices=np.argmax(preds, axis=1)

preds_labels = (train_generator_299.class_indices)

labels = dict((v,k) for k,v in preds_labels.items())

final_predictions = [labels[k] for k in preds_class_indices]



submission = pd.read_csv(os.path.join(CSV_PATH, 'sample_submission.csv'))

submission["class"] = final_predictions

submission.to_csv("submission.csv", index=False)

submission.head()
# 0.34, 0.34, 0.32가 제일 좋음.

preds = preds_list[0]

preds_class_indices=np.argmax(preds, axis=1)

preds_labels = (train_generator_299.class_indices)

labels = dict((v,k) for k,v in preds_labels.items())

final_predictions = [labels[k] for k in preds_class_indices]



submission = pd.read_csv(os.path.join(CSV_PATH, 'sample_submission.csv'))

submission["class"] = final_predictions

submission.to_csv("submission2.csv", index=False)

submission.head()
# 0.34, 0.34, 0.32가 제일 좋음.

preds = preds_list[1]

preds_class_indices=np.argmax(preds, axis=1)

preds_labels = (train_generator_299.class_indices)

labels = dict((v,k) for k,v in preds_labels.items())

final_predictions = [labels[k] for k in preds_class_indices]



submission = pd.read_csv(os.path.join(CSV_PATH, 'sample_submission.csv'))

submission["class"] = final_predictions

submission.to_csv("submission3.csv", index=False)

submission.head()
# 0.34, 0.34, 0.32가 제일 좋음.

preds = (preds_list[0] * 0.8) + (preds_list[1] * 0.2)

preds_class_indices=np.argmax(preds, axis=1)

preds_labels = (train_generator_299.class_indices)

labels = dict((v,k) for k,v in preds_labels.items())

final_predictions = [labels[k] for k in preds_class_indices]



submission = pd.read_csv(os.path.join(CSV_PATH, 'sample_submission.csv'))

submission["class"] = final_predictions

submission.to_csv("submission4.csv", index=False)

submission.head()
# 0.34, 0.34, 0.32가 제일 좋음.

preds = (preds_list[0] * 0.2) + (preds_list[1] * 0.8)

preds_class_indices=np.argmax(preds, axis=1)

preds_labels = (train_generator_299.class_indices)

labels = dict((v,k) for k,v in preds_labels.items())

final_predictions = [labels[k] for k in preds_class_indices]



submission = pd.read_csv(os.path.join(CSV_PATH, 'sample_submission.csv'))

submission["class"] = final_predictions

submission.to_csv("submission5.csv", index=False)

submission.head()