# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pydicom



import sys

from matplotlib import cm

from matplotlib import pyplot as plt

from matplotlib import patches as patches

import glob

sys.path.insert(0, '../input')
from mask_functions import rle2mask
print(os.listdir("../input/sample images"))
def showDicomTags(dataset):

    return dataset.dir()

for file_path in glob.glob('../input/sample images/*.dcm'):

    dataset = pydicom.dcmread(file_path)

    tags=showDicomTags(dataset)

    print(type(tags))

    print(tags)

    

    break

    
IMAGE_PATH = '../input/sample images/'

IMAGE_MEDIA_TYPE = '.dcm'

IMAGE_SIZE = 1024

train_rle_sample = pd.read_csv(IMAGE_PATH + 'train-rle-sample.csv', header=None, index_col=0)
def show_dicom_info(dataset):

    listTags=list(showDicomTags(dataset))

    pat_name = dataset.PatientName

    display_name = pat_name.family_name + ", " + pat_name.given_name

    print(display_name)

    print(dataset.AccessionNumber)

    print(dataset.BitsAllocated)

    print(dataset.BitsStored)

    print(dataset.BodyPartExamined)

    print(dataset.Columns)

    print(dataset.ConversionType)

    print(dataset.HighBit)

    print(dataset.InstanceNumber)

    print(dataset.LossyImageCompression)

    print(dataset.LossyImageCompressionMethod)

    print(dataset.Modality)

    print(dataset.PatientAge)

    print(dataset.PatientBirthDate)

    print(dataset.PatientID)

    print(dataset.PatientName)

    print(dataset.PatientOrientation)

    print(dataset.PatientSex)

    print(dataset.PhotometricInterpretation)

    #print(dataset.PixelData)

    print(dataset.PixelRepresentation)

    print(dataset.PixelSpacing)

    print(dataset.ReferringPhysicianName)

    print(dataset.Rows)

    print(dataset.SOPClassUID)

    print(dataset.SOPInstanceUID)

    print(dataset.SamplesPerPixel)

    print(dataset.SeriesDescription)

    print(dataset.SeriesInstanceUID)

    print(dataset.SeriesNumber)

    print(dataset.SpecificCharacterSet)

    print(dataset.StudyDate)

    print(dataset.StudyID)

    print(dataset.StudyInstanceUID)

    print(dataset.StudyTime)

    print(dataset.ViewPosition)





    if 'PixelData' in dataset:

        rows = int(dataset.Rows)

        cols = int(dataset.Columns)

        print("Image size.......: {rows:d} x {cols:d}, {size:d} bytes".format(

            rows=rows, cols=cols, size=len(dataset.PixelData)))



def plot_pixel_array(dataset, figsize=(10,10)):

    plt.figure(figsize=figsize)

    plt.imshow(dataset.pixel_array, cmap=plt.cm.bone)

    plt.show()

def plot_with_mask_and_bbox(dataset, mask_encoded, figsize=(20,10)):

    mask_decoded = rle2mask(mask_encoded, 1024, 1024).T

    fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(20,10))

    rmin, rmax, cmin, cmax = bounding_box(mask_decoded)

    patch = patches.Rectangle((cmin,rmin),cmax-cmin,rmax-rmin,linewidth=1,edgecolor='r',facecolor='none')

    ax[0].imshow(dataset.pixel_array, cmap=plt.cm.bone)

    ax[0].imshow(mask_decoded, alpha=0.3, cmap="Reds")

    ax[0].add_patch(patch)

    ax[0].set_title('With Mask')



    patch = patches.Rectangle((cmin,rmin),cmax-cmin,rmax-rmin,linewidth=1,edgecolor='r',facecolor='none')

    ax[1].imshow(dataset.pixel_array, cmap=plt.cm.bone)

    ax[1].add_patch(patch)

    ax[1].set_title('Without Mask')

    plt.show()

def show_dcm_info(dataset, image_name):

    print("Image............:", image_name)

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

def bounding_box(img):

    rows = np.any(img, axis=1)

    cols = np.any(img, axis=0)

    rmin, rmax = np.where(rows)[0][[0, -1]]

    cmin, cmax = np.where(cols)[0][[0, -1]]



    return rmin, rmax, cmin, cmax

def show_image(image_name):

    dataset = pydicom.dcmread(IMAGE_PATH + image_name + IMAGE_MEDIA_TYPE)

    show_dcm_info(dataset, image_name)

    

    mask_encoded = train_rle_sample.loc[image_name].values[0]

    if mask_encoded == '-1':    

        plot_pixel_array(dataset)

    else:

        plot_with_mask_and_bbox(dataset, mask_encoded)

for file_path in glob.glob('../input/sample images/*.dcm'):

    dataset = pydicom.dcmread(file_path)

    show_dicom_info(dataset)

    break

    

    
for file_path in glob.glob('../input/sample images/*.dcm')[0:3]:

    dataset = pydicom.dcmread(file_path)

    #show_dicom_info(dataset)

    plot_pixel_array(dataset)

    

    

show_image('1.2.276.0.7230010.3.1.4.8323329.4982.1517875185.837576')