#!pip install dicom
#import dicom 

import pydicom as dicom

import pandas as pd

import glob

import os

from tqdm import tqdm

import cv2

from PIL import Image

outdir = '../input/train_jpg/'

folder_path = "../input/siim-acr-pneumothorax-segmentation-data/pneumothorax/dicom-images-train/"

images_path = os.listdir(folder_path)





def dictify(ds):

    output = dict()

    for elem in ds:

        if elem.VR != 'SQ': 

            output[elem.tag] = elem.value

        else:

            output[elem.tag] = [dictify(item) for item in elem]

    return output



metadata_train = []

image_names_train = []

for i in tqdm(range(len(images_path))):

    t = os.listdir(folder_path+images_path[i])

    for j in range(len(t)):

        img_path = os.listdir(folder_path+images_path[i]+'/'+t[j])

        for k in range(len(img_path)):

            #ds = dicom.dcmread(folder_path+images_path[i]+'/'+t[j]+'/'+img_path[k])

            ds1 = dicom.dcmread(folder_path+images_path[i]+'/'+t[j]+'/'+img_path[k], stop_before_pixels=True)

            metadata_train.append(dictify(ds1))

            #img = ds.pixel_array

            #im = Image.fromarray(img)

            #im.save(img_path[k].replace('.dcm','.png'))

            #image_names_train.append(img_path[k].replace('.dcm','.png'))
folder_path_test = "../input/siim-acr-pneumothorax-segmentation-data/pneumothorax/dicom-images-test/"

images_path_test = os.listdir(folder_path_test)



metadata_test = []

image_names_test = []

for i in tqdm(range(len(images_path_test))):

    t = os.listdir(folder_path_test+images_path_test[i])

    for j in range(len(t)):

        img_path = os.listdir(folder_path_test+images_path_test[i]+'/'+t[j])

        for k in range(len(img_path)):

            #ds = dicom.dcmread(folder_path_test+images_path_test[i]+'/'+t[j]+'/'+img_path[k])

            ds1 = dicom.dcmread(folder_path_test+images_path_test[i]+'/'+t[j]+'/'+img_path[k], stop_before_pixels=True)

            metadata_test.append(dictify(ds1))

            #img = ds.pixel_array

            #im = Image.fromarray(img)

            #im.save(img_path[k].replace('.dcm','.png'))

            #image_names_test.append(img_path[k].replace('.dcm','.png'))
sample = dicom.dcmread('../input/siim-acr-pneumothorax-segmentation-data/pneumothorax/dicom-images-test/1.2.276.0.7230010.3.1.2.8323329.6536.1517875198.802171/1.2.276.0.7230010.3.1.3.8323329.6536.1517875198.802170/1.2.276.0.7230010.3.1.4.8323329.6536.1517875198.802172.dcm')

print(sample)
columns_list = ['Specific Character Set','SOP Class UID','SOP Instance UID','Study Date','Study Time', 'Accession Number','Modality','Conversion Type',"Referring Physician's Name",

               'Series Description',"Patient's Name",'Patient ID',"Patient's Birth Date","Patient's Sex","Patient's Age",'Body Part Examined','View Position','Study Instance UID',

               'Series Instance UID', 'Study ID', 'Series Number', 'Instance Number','Patient Orientation','Samples per Pixel','Photometric Interpretation','Rows', 'Columns',

               'Pixel Spacing','Bits Allocated','Bits Stored','High Bit','Pixel Representation','Lossy Image Compression','Lossy Image Compression Method']
train = pd.DataFrame(metadata_train)

test = pd.DataFrame(metadata_test)

train.columns = columns_list

test.columns = columns_list

#train_image = pd.DataFrame({'img':image_names_train})

#test_image = pd.DataFrame({'img':image_names_test})
train.head()
#!pip install pandas-profiling
import pandas_profiling
pandas_profiling.ProfileReport(train)
train.to_csv('meta_train.csv', index=False)

test.to_csv('meta_test.csv', index=False)



#train_image.to_csv('train_img.csv', index=False)

#test_image.to_csv('test_img.csv', index=False)