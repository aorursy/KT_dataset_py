import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

import os, glob

import pydicom as dcm

print('Loaded in libraries!')
PATH = "../input/rsna-str-pulmonary-embolism-detection/"



train_df = pd.read_csv(PATH + "train.csv")

test_df = pd.read_csv(PATH + "test.csv")



TRAIN_PATH = PATH + "train/"

TEST_PATH = PATH + "test/"

sub = pd.read_csv(PATH + "sample_submission.csv")

train_image_file_paths = glob.glob(TRAIN_PATH + '/*/*/*.dcm')

test_image_file_paths = glob.glob(TEST_PATH + '/*/*/*.dcm')



print(f'Train dataframe shape  :{train_df.shape}')

print(f'Test dataframe shape   :{test_df.shape}')



print(f'Number of train images : {len(train_image_file_paths)}')

print(f'Number of test images  : {len(test_image_file_paths)}')
# Function to take care of teh translation and windowing. 

def window_image(img, window_center,window_width, intercept, slope, rescale=True):

    img = (img*slope +intercept) #for translation adjustments given in the dicom file. 

    img_min = window_center - window_width//2 #minimum HU level

    img_max = window_center + window_width//2 #maximum HU level

    img[img<img_min] = img_min #set img_min for all HU levels less than minimum HU level

    img[img>img_max] = img_max #set img_max for all HU levels higher than maximum HU level

    if rescale: 

        img = (img - img_min) / (img_max - img_min)*255.0 

    return img

    

def get_first_of_dicom_field_as_int(x):

    #get x[0] as in int is x is a 'pydicom.multival.MultiValue', otherwise get int(x)

    if type(x) == dcm.multival.MultiValue: return int(x[0])

    else: return int(x)

    

def get_windowing(data):

    dicom_fields = [data[('0028','1050')].value, #window center

                    data[('0028','1051')].value, #window width

                    data[('0028','1052')].value, #intercept

                    data[('0028','1053')].value] #slope

    return [get_first_of_dicom_field_as_int(x) for x in dicom_fields]
def view_images(files, title = '', aug = None, windowing = True):

    width = 2

    height = 2

    fig, axs = plt.subplots(height, width, figsize=(15,15))

    

    for im in range(0, height * width):

        data = dcm.dcmread(files[im])

        image = data.pixel_array

        window_center , window_width, intercept, slope = get_windowing(data)

        if windowing:

            output = window_image(image, window_center, window_width, intercept, slope, rescale = False)

        else:

            output = image

        i = im // width

        j = im % width

        axs[i,j].imshow(output, cmap=plt.cm.gray) 

        axs[i,j].axis('off')

        

    plt.suptitle(title)

    plt.show()
view_images(train_image_file_paths[3200:], 'Images with Windowing')
view_images(train_image_file_paths[3200:], title = 'Images with Windowing', windowing=False)
data = dcm.dcmread(train_image_file_paths[3203])

image = data.pixel_array

window_center , window_width, intercept, slope = get_windowing(data)

output = window_image(image, window_center, window_width, intercept, slope, rescale = False)

f, axarr = plt.subplots(1,2, figsize=(15,10))

axarr[0].imshow(image, cmap='gray')

axarr[1].imshow(output, cmap = 'gray')
data = dcm.dcmread(train_image_file_paths[3203])

image = data.pixel_array

window_center , window_width, intercept, slope = get_windowing(data)



print(window_center , window_width, intercept, slope)

from ipywidgets import interact

def int_print(window_center , window_width=500, intercept=-1024, slope=1):

    output = window_image(image, window_center, window_width, intercept, slope, rescale = False)

    f, axarr = plt.subplots(1,2, figsize=(15,10))

    axarr[0].imshow(image, cmap='gray')

    axarr[1].imshow(output, cmap = 'gray')

    
interact(int_print, window_center= 1000)
import pandas as pd



PATH = "../input/rsna-str-pulmonary-embolism-detection/"

train = pd.read_csv(PATH + "train.csv")

sub = pd.read_csv(PATH + "sample_submission.csv")



feats = list(train.columns[3:5])+list(train.columns[8:12])+list(train.columns[13:17])

means = train[feats].mean().to_dict()





sub['label'] = 0.28

for feat in means.keys():

    sub.loc[sub.id.str.contains(feat, regex=False), 'label'] = means[feat]

    

sub.to_csv('submission.csv', index = False)