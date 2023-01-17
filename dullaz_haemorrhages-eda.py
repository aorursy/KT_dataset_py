import os

import json



import cv2

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import pydicom

from keras import layers

from keras.applications import DenseNet121

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import Callback, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from keras.preprocessing.image import ImageDataGenerator

from keras.initializers import Constant

from keras.utils import Sequence

from keras.models import Sequential

from keras.optimizers import Adam

from keras.models import Model, load_model

from keras.layers import GlobalAveragePooling2D, Dense, Activation, concatenate, Dropout

from keras.initializers import glorot_normal, he_normal

from keras.regularizers import l2

from tensorflow.python.ops import array_ops

from tqdm import tqdm

from sklearn.model_selection import train_test_split, StratifiedKFold

from keras import backend as K

import tensorflow as tf
BASE_PATH = '../input/rsna-intracranial-hemorrhage-detection/rsna-intracranial-hemorrhage-detection/'

TRAIN_DIR = 'stage_2_train/'

TEST_DIR = 'stage_2_test/'
train_df = pd.read_csv(BASE_PATH + 'stage_2_train.csv')

#sub_df = pd.read_csv(BASE_PATH + 'stage_1_sample_submission.csv')



train_df['id'] = train_df['ID'].apply(lambda st: "ID_" + st.split('_')[1])

train_df['subtype'] = train_df['ID'].apply(lambda st: st.split('_')[2])

train_df['filename'] = train_df['ID'].apply(lambda st: "ID_" + st.split('_')[1] + ".png")

train_df['type'] = train_df['ID'].apply(lambda st: st.split('_')[2])



print(train_df.shape)

train_df.head()
pivot_df = train_df

pivot_df = pivot_df[['Label', 'filename', 'type']].drop_duplicates().pivot(index='filename', columns='type', values='Label').reset_index()



print(pivot_df.shape)

pivot_df.head()
def map_to_gradient(grey_img):

    rainbow_img = np.zeros((grey_img.shape[0], grey_img.shape[1], 3))

    rainbow_img[:, :, 0] = np.clip(4 * grey_img - 2, 0, 1.0) * (grey_img > 0) * (grey_img <= 1.0)

    rainbow_img[:, :, 1] =  np.clip(4 * grey_img * (grey_img <=0.75), 0,1) + np.clip((-4*grey_img + 4) * (grey_img > 0.75), 0, 1)

    rainbow_img[:, :, 2] = np.clip(-4 * grey_img + 2, 0, 1.0) * (grey_img > 0) * (grey_img <= 1.0)

    return rainbow_img



def rainbow_window(dcm):

    grey_img = window_image(dcm, 40, 80)

    return map_to_gradient(grey_img)



def sigmoid_window(dcm, window_center, window_width, U=1.0, eps=(1.0 / 255.0)):

    _, _, intercept, slope = get_windowing(dcm)

    img = dcm.pixel_array * slope + intercept

    ue = np.log((U / eps) - 1.0)

    W = (2 / window_width) * ue

    b = ((-2 * window_center) / window_width) * ue

    z = W * img + b

    img = U / (1 + np.power(np.e, -1.0 * z))

    img = (img - np.min(img)) / (np.max(img) - np.min(img))

    return img



def sigmoid_bsb_window(dcm):

    brain_img = sigmoid_window(dcm, 40, 80)

    subdural_img = sigmoid_window(dcm, 80, 200)

    bone_img = sigmoid_window(dcm, 600, 2000)

    

    bsb_img = np.zeros((brain_img.shape[0], brain_img.shape[1], 3))

    bsb_img[:, :, 0] = brain_img

    bsb_img[:, :, 1] = subdural_img

    bsb_img[:, :, 2] = bone_img

    return bsb_img



def window_image(dcm, window_center, window_width):

    _, _, intercept, slope = get_windowing(dcm)

    img = dcm.pixel_array * slope + intercept

    img_min = window_center - window_width // 2

    img_max = window_center + window_width // 2

    img[img < img_min] = img_min

    img[img > img_max] = img_max

    img = (img - np.min(img)) / (np.max(img) - np.min(img))

    return img



def bsb_window(dcm):

    brain_img = window_image(dcm, 40, 80)

    subdural_img = window_image(dcm, 80, 200)

    bone_img = window_image(dcm, 600, 2000)

    

    bsb_img = np.zeros((brain_img.shape[0], brain_img.shape[1], 3))

    bsb_img[:, :, 0] = brain_img

    bsb_img[:, :, 1] = subdural_img

    bsb_img[:, :, 2] = bone_img

    return bsb_img

    

def get_first_of_dicom_field_as_int(x):

    #get x[0] as in int is x is a 'pydicom.multival.MultiValue', otherwise get int(x)

    if type(x) == pydicom.multival.MultiValue:

        return int(x[0])

    else:

        return int(x)



def get_windowing(data):

    dicom_fields = [data[('0028','1050')].value, #window center

                    data[('0028','1051')].value, #window width

                    data[('0028','1052')].value, #intercept

                    data[('0028','1053')].value] #slope

    return [get_first_of_dicom_field_as_int(x) for x in dicom_fields]
labels = ["epidural","intraparenchymal","intraventricular","subarachnoid","subdural"]

EDA = pivot_df[pivot_df["any"]==0].sample(n=1)

EDA['title'] = "any"

for label in labels:

    d = pivot_df[pivot_df[label] == 1].sample(n=1)

    d['title'] = label

    EDA = EDA.append(d,ignore_index=True)

EDA
def process_dcm(dcm,type="WINDOW"):

    if type == "WINDOW":

        window_center , window_width, intercept, slope = get_windowing(dcm)

        win_img = window_image(dcm, window_center, window_width)

        return win_img

    elif type == "SIGMOID":

        window_center , window_width, intercept, slope = get_windowing(dcm)

        test_img = dcm.pixel_array

        win_img = sigmoid_window(dcm, window_center, window_width)

        return win_img

    elif type == "BSB":

        win_img = bsb_window(dcm)

        return win_img

    elif type == "SIGMOID_BSB":

        return sigmoid_bsb_window(dcm)

    elif type == "GRADIENT":

        win_img = rainbow_window(dcm)

        return win_img

        
if not os.path.exists('/kaggle/working/windows/'):

    os.makedirs('/kaggle/working/windows/')

t = ["WINDOW","SIGMOID","BSB","SIGMOID_BSB","GRADIENT"]

#t = ["GRADIENT"]

for window in t:

    for index,row in EDA.iterrows():

        f,ax = plt.subplots(1,1,figsize=(10,10))

        file = row["filename"]

        dcm = pydicom.dcmread(BASE_PATH + TRAIN_DIR + file.split(".")[0] + ".dcm")

        img = (process_dcm(dcm,window) * 255.0).astype(np.uint8)

        ax.set_title("Type: " + row['title'] + "  Window: " + window )

        ax.imshow(img,cmap="bone")

        plt.savefig("/kaggle/working/" + window + "_" + row["title"] + ".png")

        plt.show()

        

    