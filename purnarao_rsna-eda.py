# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

import glob, pylab, pandas as pd

import pydicom, numpy as np

from os import listdir

from os.path import isfile, join

import matplotlib.pylab as plt

import os

import seaborn as sns

import cv2



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/rsna-intracranial-hemorrhage-detection/stage_1_train.csv')

label = data.Label.values

data = data.ID.str.rsplit("_", n=1, expand=True)

data.loc[:,'label'] = label

data.columns = ['id','subtype','label']
data.head(10)
train_images_dir = '../input/rsna-intracranial-hemorrhage-detection/stage_1_train_images/'

test_images_dir = '../input/rsna-intracranial-hemorrhage-detection/stage_1_test_images/'

# train_images = [f for f in listdir(train_images_dir) if isfile(join(train_images_dir, f))]

# test_images = [f for f in listdir(test_images_dir) if isfile(join(test_images_dir, f))]
print('Number of train images:', len(train_images))

print('Number of test images:', len(test_images))
# fig=plt.figure(figsize=(15, 10))

# columns = 3; rows = 1

# for i in range(1, columns*rows +1):

#     ds = pydicom.dcmread(train_images_dir + train_images[i])

#     print("****************************")

#     print("For the image {} the window center is {} and length is {} ".format(train_images[i] , ds[('0028','1050')].value, ds[('0028','1051')].value))

#     print("Pixel Spacing is as : ", ds[('0028','0030')].value)

#     fig.add_subplot(rows, columns, i)

#     plt.imshow(ds.pixel_array, cmap=plt.cm.bone)

#     fig.add_subplot
# pixel_space_X = []

# pixel_space_Y = []

# over = []

# under = []

# for train_image in (train_images[:1500]):

#     ds = pydicom.dcmread(train_images_dir + train_image)

#     pixel_space_X.append(float(ds[('0028','0030')].value[0]))

#     if float(ds[('0028','0030')].value[0]) > 0.50:

#         over.append(train_image)

#         print(ds[('0028','0030')].value[1])

#     if float(ds[('0028','0030')].value[0]) < 0.45:

#         under.append(train_image)

#     pixel_space_Y.append(float(ds[('0028','0030')].value[1]))
# #print(over)

# ds = pydicom.dcmread(train_images_dir + over[1])

# plt.imshow(ds.pixel_array,cmap=plt.cm.bone)

# print(ds[('0028','0030')].value[0])
# ds = pydicom.dcmread(train_images_dir + under[6])

# plt.imshow(ds.pixel_array,cmap=plt.cm.bone)

# print(ds[('0028','0030')].value[0])
# fig, ax = plt.subplots(1,2,figsize=(20,5))

# sns.distplot(pixel_space_X, ax=ax[0], color="Blue", kde=False)

# ax[0].set_title("Pixel spacing width \n distribution")

# ax[0].set_ylabel("Frequency given 1000 images")

# sns.distplot(pixel_space_Y, ax=ax[1], color="Green", kde=False)

# ax[1].set_title("Pixel spacing height \n distribution");

# ax[1].set_ylabel("Frequency given 1000 images");
def get_first_of_dicom_field_as_int(x):

    #get x[0] as in int is x is a 'pydicom.multival.MultiValue', otherwise get int(x)

    if type(x) == pydicom.multival.MultiValue:

        return int(x[0])

    else:

        return int(x)
def get_windowing(data):

    dicom_fields = [data[('0028','1050')].value,  #window center

                    data[('0028','1051')].value, #window width

                    data[('0028','1052')].value, #intercept

                    data[('0028','1053')].value] #slope

    return [get_first_of_dicom_field_as_int(x) for x in dicom_fields]
def window_image(img, window_center,window_width, intercept, slope):



    img = (img*slope +intercept)

    img_min = window_center - window_width//2

    img_max = window_center + window_width//2

    img[img<img_min] = 0

    img[img>img_max] = 255

    return img
def view_images(images, title = '', aug = None):

    width = 5

    height = 2

    fig, axs = plt.subplots(height, width, figsize=(15,5))

    

    for im in range(0, height * width):

        ''''

        image = pydicom.read_file(os.path.join(train_images_dir,'ID_'+images[im]+ '.dcm')).pixel_array

        i = im // width

        j = im % width

        axs[i,j].imshow(image, cmap=plt.cm.bone) 

        axs[i,j].axis('off')'''''

        

        data = pydicom.read_file(os.path.join(train_images_dir,'ID_'+images[im]+ '.dcm'))

        image = data.pixel_array

        window_center , window_width, intercept, slope = get_windowing(data)

        image_windowed = window_image(image, window_center, window_width, intercept, slope)

        i = im // width

        j = im % width

        axs[i,j].imshow(image_windowed, cmap=plt.cm.bone)

        #cv2.imwrite(images[im] + '.png',image_windowed)

        axs[i,j].axis('off')

        

        

    plt.suptitle(title)

    plt.show()

        
def save_and_resize(filenames, load_dir):    

    save_dir = 'sample_jpg_files/'

    if not os.path.exists(save_dir):

        os.makedirs(save_dir)



    for filename in filenames:

        path = load_dir + filename

        new_path = save_dir + filename.replace('.dcm', '.png')

        

        dcm = pydicom.dcmread(path)

        window_center , window_width, intercept, slope = get_windowing(dcm)

        img = dcm.pixel_array

        img = window_image(img, window_center, window_width, intercept, slope)

        plt.imshow(img)

#         ////////

        #resized = cv2.resize(img, (512, 512))

        res = cv2.imwrite(new_path, img)
train = pd.read_csv('../input/rsna-intracranial-hemorrhage-detection/stage_1_train.csv')

train['Sub_type'] = train['ID'].str.split("_", n = 3, expand = True)[2]

train['PatientID'] = train['ID'].str.split("_", n = 3, expand = True)[1]
view_images(train[:20].PatientID.values, title = 'Images of hemorrhage subarachnoid')
file_names = os.listdir(train_images_dir)

save_and_resize(filenames=file_names[0:20], load_dir = train_images_dir)
os.listdir("sample_jpg_files/")