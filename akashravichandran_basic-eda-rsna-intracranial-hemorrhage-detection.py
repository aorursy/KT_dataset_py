import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import warnings

warnings.filterwarnings('ignore')



print(os.listdir("../input/rsna-intracranial-hemorrhage-detection"))

import glob



import pydicom



from matplotlib import cm

from matplotlib import pyplot as plt



from keras.models import Model

from keras.layers import Input

from keras.layers.convolutional import Conv2D, Conv2DTranspose

from keras.layers.pooling import MaxPooling2D

from keras.layers.merge import concatenate

from keras import backend as K



import tensorflow as tf



from tqdm import tqdm_notebook

# Any results you write to the current directory are saved as output.
path = "../input/rsna-intracranial-hemorrhage-detection"

dataset = pydicom.dcmread("../input/rsna-intracranial-hemorrhage-detection/stage_1_train_images/ID_231d901c1.dcm")

print("Finding the details that can be fetched from a dcm file")

print("")

print(dataset)
# https://www.kaggle.com/jesperdramsch/intro-chest-xray-dicom-viz-u-nets-full-data



def show_dcm_info(dataset):

    print("Storage type.....:", dataset.SOPInstanceUID)

    print()



    print("Photometric.........:", dataset.PhotometricInterpretation)

    print("Patient id..........:", dataset.PatientID)

    print("Modality............:", dataset.Modality)

    print("Image Position......:", dataset.ImagePositionPatient)

    print("Image Orient........:", dataset.ImageOrientationPatient)

    

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
show_dcm_info(dataset)

plot_pixel_array(dataset)
start = 5   # Starting index of images

num_img = 10 # Total number of images to show



fig, ax = plt.subplots(nrows=1, ncols=num_img, sharey=True, figsize=(num_img*10,10))

for q, file_path in enumerate(glob.glob('../input/rsna-intracranial-hemorrhage-detection/stage_1_train_images/*.dcm')[start:start+num_img]):

    dataset = pydicom.dcmread(file_path)

    #show_dcm_info(dataset)

    ax[q].imshow(dataset.pixel_array, cmap=plt.cm.bone)
train_df = pd.read_csv(path + '/stage_1_train.csv')

test_df = pd.read_csv(path + '/stage_1_sample_submission.csv')
print('Train -', len(train_df))

print('Test -', len(test_df))
train_df.head(12)
train_df[['PID','Test']] = train_df.ID.str.rsplit("_", n=1, expand=True)
epidural = train_df[train_df.Test == 'epidural']

intraparenchymal = train_df[train_df.Test == 'intraparenchymal']

intraventricular = train_df[train_df.Test == 'intraventricular']

subarachnoid = train_df[train_df.Test == 'subarachnoid']

subdural = train_df[train_df.Test == 'subdural']

anyy = train_df[train_df.Test == 'any']
print('EPIDURAL')

display(epidural['Label'].value_counts())

print('INTRAPARENCHYMAL')

display(intraparenchymal['Label'].value_counts())

print('INTRAVENTRICULAR')

display(intraventricular['Label'].value_counts())

print('SUBARACHNOID')

display(subarachnoid['Label'].value_counts())

print('SUBDURAL')

display(subdural['Label'].value_counts())

print("ANY")

display(anyy['Label'].value_counts())