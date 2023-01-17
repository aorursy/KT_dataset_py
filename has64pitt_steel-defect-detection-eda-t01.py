# import basics

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

from glob import glob



# import plotting

from matplotlib import pyplot as plt

import matplotlib.patches as patches

import matplotlib

import seaborn as sns



# import image manipulation

from PIL import Image

import imageio



! pip install --upgrade imgaug



# import data augmentation

import imgaug as ia

from imgaug import augmenters as iaa

# import segmentation maps from imgaug

from imgaug.augmentables.segmaps import SegmentationMapOnImage

import imgaug.imgaug
# set paths to train and test image datasets

TRAIN_PATH = '../input/severstal-steel-defect-detection/train_images/'

TEST_PATH = '../input/severstal-steel-defect-detection/test_images/'



# load dataframe with train labels

train_df = pd.read_csv('../input/severstal-steel-defect-detection/train.csv')

train_fns = sorted(glob(TRAIN_PATH + '*.jpg'))

test_fns = sorted(glob(TEST_PATH + '*.jpg'))



print('There are {} images in the train set.'.format(len(train_fns)))

print('There are {} images in the test set.'.format(len(test_fns)))
# plotting a pie chart which demonstrates train and test sets

labels = 'Train', 'Test'

sizes = [len(train_fns), len(test_fns)]

explode = (0, 0.1)



fig, ax = plt.subplots(figsize=(6, 6))

ax.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)

ax.axis('equal')

ax.set_title('Train and Test Sets')



plt.show()
train_df.head(10)
print('There are {} rows with empty segmentation maps.'.format(len(train_df) - train_df.EncodedPixels.count()))
# plotting a pie chart

labels = 'Non-empty', 'Empty'

sizes = [train_df.EncodedPixels.count(), len(train_df) - train_df.EncodedPixels.count()]

explode = (0, 0.1)



fig, ax = plt.subplots(figsize=(6, 6))

ax.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)

ax.axis('equal')

ax.set_title('Non-empty and Empty Masks')



plt.show()
# split column

split_df = train_df["ImageId_ClassId"].str.split("_", n = 1, expand = True)



# add new columns to train_df

train_df['Image'] = split_df[0]

train_df['Label'] = split_df[1]



# check the result

train_df.head()
defect1 = train_df[train_df['Label'] == '1'].EncodedPixels.count()

defect2 = train_df[train_df['Label'] == '2'].EncodedPixels.count()

defect3 = train_df[train_df['Label'] == '3'].EncodedPixels.count()

defect4 = train_df[train_df['Label'] == '4'].EncodedPixels.count()



labels_count = train_df.groupby('Image').count()['EncodedPixels']

no_defects = len(labels_count) - labels_count.sum()



print('There are {} defect1 images'.format(defect1))

print('There are {} defect2 images'.format(defect2))

print('There are {} defect3 images'.format(defect3))

print('There are {} defect4 images'.format(defect4))

print('There are {} images with no defects'.format(no_defects))
# plotting a pie chart

labels = 'Defect 1', 'Defect 2', 'Defect 3', 'Defect 4', 'No defects'

sizes = [defect1, defect2, defect3, defect4, len(train_fns) - defect1 - defect2 - defect3 - defect4]



fig, ax = plt.subplots(figsize=(6, 6))

ax.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)

ax.axis('equal')

ax.set_title('Defect Types')



plt.show()
labels_per_image = train_df.groupby('Image')['EncodedPixels'].count()

print('The mean number of labels per image is {}'.format(labels_per_image.mean()))
fig, ax = plt.subplots(figsize=(6, 6))

ax.hist(labels_per_image)

ax.set_title('Number of Labels per Image')
def get_image_sizes(train = True):

    '''

    Function to get sizes of images from test and train sets.

    INPUT:

        train - indicates whether we are getting sizes of images from train or test set

    '''

    if train:

        path = TRAIN_PATH

    else:

        path = TEST_PATH

        

    widths = []

    heights = []

    

    images = sorted(glob(path + '*.jpg'))

    

    max_im = Image.open(images[0])

    min_im = Image.open(images[0])

        

    for im in range(0, len(images)):

        image = Image.open(images[im])

        width, height = image.size

        

        if len(widths) > 0:

            if width > max(widths):

                max_im = image



            if width < min(widths):

                min_im = image



        widths.append(width)

        heights.append(height)

        

    return widths, heights, max_im, min_im
# get sizes of images from test and train sets

train_widths, train_heights, max_train, min_train = get_image_sizes(train = True)

test_widths, test_heights, max_test, min_test = get_image_sizes(train = False)



print('Maximum width for training set is {}'.format(max(train_widths)))

print('Minimum width for training set is {}'.format(min(train_widths)))

print('Maximum height for training set is {}'.format(max(train_heights)))

print('Minimum height for training set is {}'.format(min(train_heights)))
print('Maximum width for test set is {}'.format(max(test_widths)))

print('Minimum width for test set is {}'.format(min(test_widths)))

print('Maximum height for test set is {}'.format(max(test_heights)))

print('Minimum height for test set is {}'.format(min(test_heights)))
# https://www.kaggle.com/titericz/building-and-visualizing-masks

def rle2maskResize(rle):

    

    # CONVERT RLE TO MASK 

    if (pd.isnull(rle))|(rle=='')|(rle=='-1'): 

        return np.zeros((256,1600) ,dtype=np.uint8)

    

    height= 256

    width = 1600

    mask= np.zeros( width*height ,dtype=np.uint8)



    array = np.asarray([int(x) for x in rle.split()])

    starts = array[0::2]-1

    lengths = array[1::2]    

    for index, start in enumerate(starts):

        mask[int(start):int(start+lengths[index])] = 1

    

    return mask.reshape( (height,width), order='F' )
def plot_mask(image_filename):

    '''

    Function to plot an image and segmentation masks.

    INPUT:

        image_filename - filename of the image (with full path)

    '''

    img_id = image_filename.split('/')[-1]

    image = Image.open(image_filename)

    train = train_df.fillna('-1')

    rle_masks = train[(train['Image'] == img_id) & (train['EncodedPixels'] != '-1')]['EncodedPixels'].values

    

    defect_types = train[(train['Image'] == img_id) & (train['EncodedPixels'] != '-1')]['Label'].values

    

    if (len(rle_masks) > 0):

        fig, axs = plt.subplots(1, 1 + len(rle_masks), figsize=(20, 3))



        axs[0].imshow(image)

        axs[0].axis('off')

        axs[0].set_title('Original Image')



        for i in range(0, len(rle_masks)):

            mask = rle2maskResize(rle_masks[i])

            axs[i + 1].imshow(image)

            axs[i + 1].imshow(mask, alpha = 0.5, cmap = "Reds")

            axs[i + 1].axis('off')

            axs[i + 1].set_title('Mask with defect #{}'.format(defect_types[i]))



        plt.suptitle('Image with defect masks')

    else:

        fig, axs = plt.subplots(figsize=(20, 3))

        axs.imshow(image)

        axs.axis('off')

        axs.set_title('Original Image without Defects')
import cv2

# thresholds and min_size for segmentation predictions

# play with them and see how LB changes

threshold_pixel = [0.5,0.5,0.5,0.5,] 

min_size = [200,1500,1500,2000] 
def post_process(probability, threshold, min_size,row):

    '''Post processing of each predicted mask, components with lesser number of pixels

    than `min_size` are ignored'''

    mask = cv2.threshold(probability, threshold, 1, cv2.THRESH_BINARY)[1]

    num_component, component = cv2.connectedComponents(mask.astype(np.uint8))

    predictions = np.zeros((256, 1600), np.float32)

    num = 0

    area=[]

    for c in range(1, num_component):

        p = (component == c)      

        if p.sum() > min_size:

            predictions[p] = 1

            num += 1

            area.append(p.sum())

            #print(p.sum())

            if(p.sum()>161320):

                print(c)

                print(p.sum())

                print(row)

    return predictions, num,area
num = np.zeros([4,1])

area = [[],[],[],[]]

for i,row in train_df.iterrows():

        #print(i%4)

        #if row['Label'] == '1':

        if(i<2000000):

            pred,numtmp,areatmp = post_process(rle2maskResize(row.iloc[1]),0,0,row)

            num[i%4] += numtmp

            area[i%4].extend(areatmp)

            

#areaarray = np.asarray(area)

#print("type 1 mean = "+ str(areaarray.mean()))

#print("type 1 std = "+ str(areaarray.std()))

atemp = area[2][0:100]

print(atemp.sort())

for i in range(4):

    print("total number of isoloated defect areas for type {} is : ".format(i+1),end=' ' )

    print(num[i]     )

    print("mean area = ", end=' ')

    print(np.asarray(area[i]).mean())

    print("std area = ", end=' ')

    print(np.asarray(area[i]).std())

    print("5% percentile area = ", end=' ')

    print(np.percentile(area[i],5))

    print("10% percentile area = ", end=' ')

    print(np.percentile(area[i],10))

    print("30% percentile area = ", end=' ')

    print(np.percentile(area[i],30))

    print("50% percentile area = ", end=' ')

    print(np.percentile(area[i],50))

    print("70% percentile area = ", end=' ')

    print(np.percentile(area[i],70))

    print("90% percentile area = ", end=' ')

    print(np.percentile(area[i],90))

    print("95% percentile area = ", end=' ')

    print(np.percentile(area[i],95))

    print()

    
# Plot Histograms and KDE plots

plt.figure(figsize=(15,7))



plt.subplot(221)

sns.distplot(area[0], kde=False, label='Defect #1',hist=True)

plt.legend()

plt.title('Mask Area Histogram : Defect #1', fontsize=15)

#plt.xlim(0, 4000)



plt.subplot(222)

sns.distplot(area[1], kde=False, label='Defect #2',hist=True)

plt.legend()

plt.title('Mask Area Histogram : Defect #2', fontsize=15)

#plt.xlim(0, 10000)



plt.subplot(223)

sns.distplot(area[2], kde=False, label='Defect #3',hist=True)

plt.legend()

plt.title('Mask Area Histogram : Defect #3', fontsize=15)

#plt.xlim(0, 25000)



plt.subplot(224)

sns.distplot(area[3], kde=False, label='Defect #4',hist=True)

plt.legend()

plt.title('Mask Area Histogram : Defect #4', fontsize=15)

#plt.xlim(0, 50000)



#plt.tight_layout()

plt.show()
# plot image with single defect

plot_mask(train_fns[0])
# plot image with defects

plot_mask("../input/severstal-steel-defect-detection/train_images/df5c68422.jpg")

plot_mask("../input/severstal-steel-defect-detection/train_images/bf740c1e2.jpg")

plot_mask("../input/severstal-steel-defect-detection/train_images/df5c68422.jpg")

plot_mask("../input/severstal-steel-defect-detection/train_images/a154fdcfd.jpg")