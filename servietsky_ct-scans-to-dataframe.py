!conda install -c conda-forge gdcm -y
import numpy as np

import pandas as pd

from tqdm.notebook import tqdm 

import gc

import glob, os

import pydicom

from PIL import Image

import matplotlib.pyplot as plt

import gdcm

import pickle

import skimage.measure

import cv2
train = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv')

test = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')



train_out = pd.DataFrame()

img_array = []

instance_array = []

img_dict = {}

for Patient in tqdm(train.Patient.unique()) :

    for filename in glob.glob('../input/osic-pulmonary-fibrosis-progression/train/'+Patient+'/*'):

#         img = skimage.measure.block_reduce(pydicom.dcmread(filename).pixel_array, (3,3), np.max)

        d = pydicom.dcmread(filename)

        instance = d.InstanceNumber

        img = cv2.resize((d.pixel_array * d.RescaleSlope + d.RescaleIntercept)/1000, (512, 512))

#         img = img.flatten() #To avoid imagepooling

        img = skimage.measure.block_reduce(img, (2,2), np.max).flatten()

        img_array.append(img)

        instance_array.append(instance)

    df_tmp = pd.DataFrame(img_array).astype('float16')

    df_tmp['Instance'] = instance_array

    df_tmp['Patient'] = Patient

    train_out = pd.concat([train_out, df_tmp])

#     img_dict[Patient] = img_array

    img_array = []

    instance_array = []

gc.collect()



for Patient in tqdm(test.Patient.unique()) :

    for filename in glob.glob('../input/osic-pulmonary-fibrosis-progression/test/'+Patient+'/*'):

#         img = skimage.measure.block_reduce(pydicom.dcmread(filename).pixel_array, (3,3), np.max)

        d = pydicom.dcmread(filename)

        instance = d.InstanceNumber

        img = cv2.resize((d.pixel_array * d.RescaleSlope + d.RescaleIntercept)/1000, (512, 512))

#         img = img.flatten() #To avoid imagepooling

        img = skimage.measure.block_reduce(img, (2,2), np.max).flatten()

        img_array.append(img)

        instance_array.append(instance)

    df_tmp = pd.DataFrame(img_array).astype('float16')

    df_tmp['Instance'] = instance_array

    df_tmp['Patient'] = Patient

    train_out = pd.concat([train_out, df_tmp])

#     img_dict[Patient] = img_array

    img_array = []

    instance_array = []

gc.collect()

        

print('Train Done')
train_out.info()
train_out.head()
train_out.to_pickle("train_out.pkl")
test_out = pd.DataFrame()

i=0

img_array = []

instance_array = []

img_dict = {}

for Patient in tqdm(test.Patient.unique()) :

    for filename in glob.glob('../input/osic-pulmonary-fibrosis-progression/test/'+Patient+'/*.dcm'):

#         img = skimage.measure.block_reduce(pydicom.dcmread(filename).pixel_array, (3,3), np.max)

        d = pydicom.dcmread(filename)

        instance = d.InstanceNumber

#         img = cv2.resize((d.pixel_array - d.RescaleIntercept) / (d.RescaleSlope * 1000), (512, 512))

        img = cv2.resize((d.pixel_array * d.RescaleSlope + d.RescaleIntercept)/1000, (512, 512))

#         img = img.flatten() #To avoid imagepooling

        img = skimage.measure.block_reduce(img, (2,2), np.max).flatten()

        img_array.append(img)

        instance_array.append(instance)

    df_tmp = pd.DataFrame(img_array).astype('float16')

    df_tmp['Instance'] = instance_array

    df_tmp['Patient'] = Patient

    test_out = pd.concat([test_out, df_tmp])

#     img_dict[Patient] = img_array

    img_array = []

    instance_array = []

    gc.collect()



print('test Done')
test_out.info()
test_out
test_out.to_pickle("test_out.pkl")


# \pd.concat([train_out, test_out], ignore_index=True).to_pickle("train_out.pkl")