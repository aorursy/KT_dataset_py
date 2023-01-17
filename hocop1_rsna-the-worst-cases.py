from glob import glob

import os

import pandas as pd

import numpy as np

import re

from PIL import Image

import seaborn as sns

import pydicom

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from tqdm import tqdm_notebook as tqdm

import cv2



#checnking the input files

# print(os.listdir("../input/rsna-intracranial-hemorrhage-detection/"))
# Load datasets

train_images_dir = '../input/rsna-intracranial-hemorrhage-detection/stage_1_train_images/'

test_images_dir = '../input/rsna-intracranial-hemorrhage-detection/stage_1_test_images/'



train = pd.read_csv('../input/rsna-intracranial-hemorrhage-detection/stage_1_train.csv')

test = pd.read_csv('../input/rsna-intracranial-hemorrhage-detection/stage_1_sample_submission.csv')



# Transform training set. Code from https://www.kaggle.com/taindow/pytorch-efficientnet-b0-benchmark

train[['ID', 'Image', 'Diagnosis']] = train['ID'].str.split('_', expand=True)

train = train[['Image', 'Diagnosis', 'Label']]

train.drop_duplicates(inplace=True)

train = train.pivot(index='Image', columns='Diagnosis', values='Label').reset_index()

train['Image'] = 'ID_' + train['Image']

train.head(10)
def _get_first_of_dicom_field_as_int(x):

    #get x[0] as in int is x is a 'pydicom.multival.MultiValue', otherwise get int(x)

    if type(x) == pydicom.multival.MultiValue:

        return int(x[0])

    else:

        return int(x)



def _get_windowing(data):

    dicom_fields = [data[('0028','1050')].value, #window center

                    data[('0028','1051')].value, #window width

                    data[('0028','1052')].value, #intercept

                    data[('0028','1053')].value] #slope

    return [_get_first_of_dicom_field_as_int(x) for x in dicom_fields]



def get_image(data, windowing=None):

    window_center, window_width, intercept, slope = windowing or _get_windowing(data)

    img = data.pixel_array

    img = (img*slope +intercept)

    img_min = window_center - window_width//2

    img_max = window_center + window_width//2

    img[img<img_min] = img_min

    img[img>img_max] = img_max

    return img
fig, ax = plt.subplots(nrows=4, ncols=4, figsize=(30,30))



for i in range(16):

    idx = train[train['any'] == 0]['Image'].iloc[i]

    data = pydicom.dcmread(train_images_dir + idx + '.dcm')

    img = get_image(data)

    ax[i//4, i%4].set_title(idx)

    ax[i//4, i%4].imshow(img, cmap=plt.cm.bone)
for n in range(6):

    many = train[train[['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural']].sum(1) == n].copy()

    print('Number of hemorrhages: {}, amount of such images: {}, fraction: {:.3f}%'.format(n, len(many), 100 * len(many) / len(train)))
many
many['Patient ID'] = many['Image'].map(lambda image: pydicom.dcmread(train_images_dir + image + '.dcm')[('0010', '0020')].repval.replace("'", ''))

many
many['Patient ID'].unique()
log = []



for i, patient in enumerate(many['Patient ID'].unique()):

    print('Patient №{}: {}'.format(i + 1, patient))

    ids = many[many['Patient ID'] == patient]['Image']

    fig, ax = plt.subplots(nrows=1, ncols=len(ids), figsize=(30,10))

    for j, idx in enumerate(ids):

        data = pydicom.dcmread(train_images_dir + idx + '.dcm')

        log.append(('Patient №{}: {}'.format(i + 1, patient), idx, data))

        img = get_image(data)

        if len(ids) == 1:

            ax.set_title(idx, fontsize=40)

            ax.imshow(img, cmap=plt.cm.bone)

        else:

            ax[j].set_title(idx, fontsize=30)

            ax[j].imshow(img, cmap=plt.cm.bone)

    plt.show()
# Get id of image of patient №3

idx = many[many['Patient ID'] == many['Patient ID'].unique()[2]]['Image'].iloc[0]

data = pydicom.dcmread(train_images_dir + idx + '.dcm')

c, w, intercept, slope = _get_windowing(data)



known_windows = [('Default window', c, w),

           ('Brain Matter window', 40, 80),

           ('Blood/subdural window', 80, 200),

           ('Soft tissue window', 40, 375),

           ('Bone window', 600, 2800),

           ('Grey-white differentiation window', 32, 8)]

fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(30,20))



for i, (window_name, window_center, window_width) in enumerate(known_windows):

    img = get_image(data, [window_center, window_width, intercept, slope])

    ax[i//3, i%3].set_title(window_name, fontsize=40)

    ax[i//3, i%3].imshow(img, cmap=plt.cm.bone)
for patient, idx, data in log:

    print(patient, 'Image', idx)

    c, w, intercept, slope = _get_windowing(data)



    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(30,20))



    for i, (window_name, window_center, window_width) in enumerate(known_windows):

        img = get_image(data, [window_center, window_width, intercept, slope])

        ax[i//3, i%3].set_title(window_name, fontsize=40)

        ax[i//3, i%3].imshow(img, cmap=plt.cm.bone)

    plt.show()