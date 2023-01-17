from glob import glob

import os

import pandas as pd

import numpy as np

import re

from PIL import Image

import seaborn as sns

from random import randrange



from albumentations import (

    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,

    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,

    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, RandomBrightnessContrast, IAAPiecewiseAffine,

    IAASharpen, IAAEmboss, Flip, OneOf, Compose, PadIfNeeded, RandomGamma

)

#checnking the input files

print(os.listdir("../input/rsna-intracranial-hemorrhage-detection/"))
#reading all dcm files into train and text

train = sorted(glob("../input/rsna-intracranial-hemorrhage-detection/stage_1_train_images/*.dcm"))

test = sorted(glob("../input/rsna-intracranial-hemorrhage-detection/stage_1_test_images/*.dcm"))

print("train files: ", len(train))

print("test files: ", len(test))



pd.reset_option('max_colwidth')
train_df = pd.read_csv('../input/rsna-intracranial-hemorrhage-detection/stage_1_train.csv')
stage_1_sample_submission = pd.read_csv('../input/rsna-intracranial-hemorrhage-detection/stage_1_sample_submission.csv') 
import pydicom

import matplotlib.pyplot as plt



#displaying the image

img = pydicom.read_file(train[0]).pixel_array

plt.imshow(img, cmap=plt.cm.bone)

plt.grid(False)



#displaying metadata

data = pydicom.dcmread(train[0])

print(data)
# visualize pie chart

labels = 'Train', 'Test'

sizes = [len(train), len(test)]

explode = (0, 0.1)  # "explode" the 2nd slice



fig, ax = plt.subplots(figsize=(7,7))

ax.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.



plt.title("Number of images in train/test sets")

plt.show()
#train_df['ID'].str.find('_', 3) 

train_df['image'] = train_df['ID'].str.slice(stop=12)

train_df['diagnosis'] = train_df['ID'].str.slice(start=13)
train_df.head(6)
train_df.groupby('diagnosis').sum().plot(kind='bar',figsize = (10, 5));

plt.title('Class counts');
image_lable = train_df.groupby('image').sum()
image_lable['Label'].value_counts().plot(kind='bar',figsize = (10, 5));

plt.title('Number of people with different class of Hemorrhage');
image_lable = train_df.query('diagnosis!="any"').groupby('image').sum()
image_lable['Label'].value_counts().plot(kind='bar',figsize = (10, 5));

plt.title('Class with labels');
TRAIN_IMG_PATH = "../input/rsna-intracranial-hemorrhage-detection/stage_1_train_images/"

TEST_IMG_PATH = "../input/rsna-intracranial-hemorrhage-detection/stage_1_test_images/"



def view_images(images, title = '', aug = None):

    width = 5

    height = 2

    fig, axs = plt.subplots(height, width, figsize=(15,5))

    

    for im in range(0, height * width):

        image = pydicom.read_file(os.path.join(TRAIN_IMG_PATH,images[im]+ '.dcm')).pixel_array

        i = im // width

        j = im % width

        axs[i,j].imshow(image, cmap=plt.cm.bone) 

        axs[i,j].axis('off')

        

    plt.suptitle(title)

    plt.show()
view_images(train_df[(train_df['diagnosis'] == 'epidural') & (train_df['Label'] == 1)][:10].image.values, title = 'Images with epidural')
view_images(train_df[(train_df['diagnosis'] == 'intraparenchymal') & (train_df['Label'] == 1)][:10].image.values, title = 'Images with intraparenchymal')
view_images(train_df[(train_df['diagnosis'] == 'intraventricular')& (train_df['Label'] == 1)][:10].image.values, title = 'Images with intraventricular')
view_images(train_df[(train_df['diagnosis'] == 'subarachnoid')& (train_df['Label'] == 1)][:10].image.values, title = 'Images with subarachnoid')
view_images(train_df[(train_df['diagnosis'] == 'subdural') & (train_df['Label'] == 1)][:10].image.values, title = 'Images with subarachnoid')
def get_image_sizes(df, train = True):

    if train:

        path = TRAIN_IMG_PATH

    else:

        path = TEST_IMG_PATH

        

    widths = []

    heights = []

    

    images = df.image.values

    #print(images)

    max_im = pydicom.read_file(os.path.join(path,images[0]+ '.dcm')).pixel_array

    min_im = pydicom.read_file(os.path.join(path,images[0]+ '.dcm')).pixel_array

        

    for im in range(0, len(images)):

        image = pydicom.read_file(os.path.join(path,images[im]+ '.dcm')).pixel_array

        

        width = image.shape[0]

        height = image.shape[1]

        

        if len(widths) > 0:

            if width > max(widths):

                max_im = image



            if width < min(widths):

                min_im = image



        widths.append(width)

        heights.append(height)

        

    return widths, heights, max_im, min_im
stage_1_sample_submission['image'] = stage_1_sample_submission['ID'].str.slice(stop=12)

stage_1_sample_submission['diagnosis'] = stage_1_sample_submission['ID'].str.slice(start=13)
stage_1_sample_submission.shape
train_df.shape
train_df_d = train_df.drop_duplicates(subset='image')
train_df_d.shape
stage_1_sample_submission_d = stage_1_sample_submission.drop_duplicates(subset='image')
stage_1_sample_submission_d.shape
train_widths, train_heights, max_train, min_train = get_image_sizes(train_df_d.sample(10000), train = True)

test_widths, test_heights, max_test, min_test = get_image_sizes(stage_1_sample_submission_d.sample(10000), train = False)
print('Maximum width for training set is {}'.format(max(train_widths)))

print('Minimum width for training set is {}'.format(min(train_widths)))

print('Maximum height for training set is {}'.format(max(train_heights)))

print('Minimum height for training set is {}'.format(min(train_heights)))
print('Maximum width for test set is {}'.format(max(test_widths)))

print('Minimum width for test set is {}'.format(min(test_widths)))

print('Maximum height for test set is {}'.format(max(test_heights)))

print('Minimum height for test set is {}'.format(min(test_heights)))
plt.figure(figsize=(14,6))

plt.subplot(121)

sns.distplot(train_widths, kde=False, label='Train Width')

sns.distplot(train_heights, kde=False, label='Train Height')

plt.legend()

plt.title('Training Image Dimension Histogram', fontsize=15)



plt.subplot(122)

sns.kdeplot(train_widths, label='Train Width')

sns.kdeplot(train_heights, label='Train Height')

plt.legend()

plt.title('Train Image Dimension KDE Plot', fontsize=15)



plt.tight_layout()

plt.show()
plt.figure(figsize=(14,6))

plt.subplot(121)

sns.distplot(test_widths, kde=False, label='Test Width')

sns.distplot(test_heights, kde=False, label='Test Height')

plt.legend()

plt.title('Test Image Dimension Histogram', fontsize=15)



plt.subplot(122)

sns.kdeplot(test_widths, label='Test Width')

sns.kdeplot(test_heights, label='Test Height')

plt.legend()

plt.title('Test Image Dimension KDE Plot', fontsize=15)



plt.tight_layout()

plt.show()
plt.axis('off')

plt.imshow(max_train, cmap=plt.cm.bone) #plot the data
plt.axis('off')

plt.imshow(min_train, cmap=plt.cm.bone) #plot the data
# get some random image indices from the training set

rand_indices = [randrange(len(train_df_d)) for x in range(0,10)]

rand_indices
def view_aug_images(train, rand_indices, aug = None, title = ''):

    width = 5

    height = 2

    counter = 0

    fig, axs = plt.subplots(height, width, figsize=(15,5))

    

    for im in rand_indices:

        image = pydicom.read_file(os.path.join(TRAIN_IMG_PATH,train.iloc[im].image+ '.dcm')).pixel_array

        if aug is not None:

            image = aug(image=np.array(image))['image']

        

        i = counter // width

        j = counter % width

        axs[i,j].imshow(image, cmap=plt.cm.bone) #plot the data

        axs[i,j].axis('off')

        

        diagnosis = train[train['image'] == train.iloc[im].image].diagnosis.values[0]

        

        axs[i,j].set_title(diagnosis)

        counter += 1



    plt.suptitle(title)

    plt.show()
view_aug_images(train_df_d, rand_indices, title = 'Original images')
aug = GaussNoise(p=1)

view_aug_images(train_df_d, rand_indices, aug, title= 'GaussNoise')
aug = RandomBrightnessContrast(brightness_limit=1, contrast_limit=1, p = 1)

view_aug_images(train_df_d, rand_indices, aug, title = 'RandomBrightnessContrast')
aug = RandomGamma(gamma_limit=[80,120], p = 1)

view_aug_images(train_df_d, rand_indices, aug, title = 'RandomGamma')
aug = GridDistortion(num_steps =5, distort_limit=[-0.3,0.3], interpolation=1, border_mode= 4, p = 1)

view_aug_images(train_df_d, rand_indices, aug, title = 'GridDistortion')
aug = OpticalDistortion(shift_limit =[-0.5, 0.5], distort_limit=[-2,2], interpolation=1, border_mode= 4, p = 1)

view_aug_images(train_df_d, rand_indices, aug, title = 'OpticalDistortion')