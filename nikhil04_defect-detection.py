# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

from glob import glob

from PIL import Image

import matplotlib.pyplot as plt

import seaborn as sns

import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.






TRAIN_PATH = '../input/severstal-steel-defect-detection/train_images/'

TEST_PATH = '../input/severstal-steel-defect-detection/test_images/'



# loading training dataset

train_df = pd.read_csv('../input/severstal-steel-defect-detection/train.csv')

train_img = sorted(glob(TRAIN_PATH + '*.jpg'))

test_img = sorted(glob(TEST_PATH + '*.jpg'))



train_df.shape
train_df.head()
#plotting the pie chart for demonstration train and test set

labels = 'Train', 'Test'

sizes = [len(train_img), len(test_img)]

explode = 0, 0.1





def make_autopct(values):

    def my_autopct(pct):

        total = sum(values)

        val = int(round(pct*total/100.0))

        return '{p:.2f}%  ({v:d})'.format(p=pct,v=val)

    return my_autopct





fig, ax = plt.subplots(figsize=(6,6))

ax.pie(sizes, explode=explode, labels=labels, autopct = make_autopct(sizes), shadow=True, startangle = 90)

ax.axis('equal')

ax.set_title('Train and Test Sets')



plt.show()
print('There are {} empty records'.format(train_df.EncodedPixels.isnull().sum()))
#plotting the pie chart for demonstration Empty and Non-Empty set

labels = 'Non-empty', 'Empty'

sizes = [train_df.EncodedPixels.count(), train_df.EncodedPixels.isnull().sum()]

explode = 0, 0.1





def make_autopct(values):

    def my_autopct(pct):

        total = sum(values)

        val = int(round(pct*total/100.0))

        return '{p:.2f}%  ({v:d})'.format(p=pct,v=val)

    return my_autopct





fig, ax = plt.subplots(figsize=(6,6))

ax.pie(sizes, explode=explode, labels=labels, autopct = make_autopct(sizes), shadow=True, startangle = 90)

ax.axis('equal')

ax.set_title('Empty and Non-Empty Sets')



plt.show()
split_df = train_df['ImageId_ClassId'].str.split('_', n=1, expand=True)

train_df['Image'] = split_df[0]

train_df['Label'] = split_df[1]



train_df.head()
labels_count = train_df.groupby('Image').count()['EncodedPixels']

labels_count
#plotting the pie chart for demonstration Defects

labels = 'Defect 1', 'Defect 2', 'Defect 3', 'Defect 4', 'No Defects'

defect_1 = train_df[train_df['Label'] == '1'].EncodedPixels.count()

defect_2 = train_df[train_df['Label'] == '2'].EncodedPixels.count()

defect_3 = train_df[train_df['Label'] == '3'].EncodedPixels.count()

defect_4 = train_df[train_df['Label'] == '4'].EncodedPixels.count()

labels_count = train_df.groupby('Image').count()['EncodedPixels']

no_defects = len(labels_count) - labels_count.sum()

sizes = [defect_1, defect_2, defect_3, defect_4, no_defects]

print('There are {} defect1 images'.format(defect_1))

print('There are {} defect2 images'.format(defect_2))

print('There are {} defect3 images'.format(defect_3))

print('There are {} defect4 images'.format(defect_4))

print('There are {} images with no defects'.format(no_defects))

explode = 0, 0.1





def make_autopct(values):

    def my_autopct(pct):

        total = sum(values)

        val = int(round(pct*total/100.0))

        return '{p:.2f}%  ({v:d})'.format(p=pct,v=val)

    return my_autopct





fig, ax = plt.subplots(figsize=(6,6))

ax.pie(sizes, labels=labels, autopct = make_autopct(sizes), shadow=True, startangle = 90)

ax.axis('equal')

ax.set_title('Defects')



plt.show()
labels_per_image = train_df.groupby('Image')['EncodedPixels'].count()

print('The mean number of labels per image is {}'.format(labels_per_image.mean()))
fig, ax = plt.subplots(figsize=(6,6))

ax.hist(labels_per_image)

ax.set_title('Number of labels per image')
def get_image_sizes(train = True):

    '''

    Funtion to get the sizes of the images

    '''

    

    if train:

        path=TRAIN_PATH

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
plot_mask(train_img[0])
# plot image without defects

plot_mask(train_img[1])
# plot image example with several defects

for image_code in train_df.Image.unique():

    if (train_df.groupby(['Image'])['EncodedPixels'].count().loc[image_code] > 1):

        plot_mask(TRAIN_PATH + image_code)

        break;
def add_mask_areas(train_df):

    masks_df = train_df.copy()

    masks_df['Area'] = 0

    

    for i, row in masks_df.iterrows():

        masks_df['Area'].loc[i] = np.sum(get_mask(i))

        

    return masks_df



def get_mask(line_id):

    # convert rle to mask

    rle = train_df.loc[line_id]['EncodedPixels']

    

    np_mask = rle2maskResize(rle)

    np_mask = np.clip(np_mask, 0, 1)

        

    return np_mask
masks_df = add_mask_areas(train_df)
# Plot Histograms and KDE plots

plt.figure(figsize=(15,7))



plt.subplot(221)

sns.distplot(masks_df[masks_df['Label'] == '1']['Area'].values, kde=False, label='Defect #1')

plt.legend()

plt.title('Mask Area Histogram : Defect #1', fontsize=15)



plt.subplot(222)

sns.distplot(masks_df[masks_df['Label'] == '2']['Area'].values, kde=False, label='Defect #2')

plt.legend()

plt.title('Mask Area Histogram: Defect #2', fontsize=15)



plt.subplot(223)

sns.distplot(masks_df[masks_df['Label'] == '3']['Area'].values, kde=False, label='Defect #3')

plt.legend()

plt.title('Mask Area Histogram : Defect #3', fontsize=15)



plt.subplot(224)

sns.distplot(masks_df[masks_df['Label'] == '4']['Area'].values, kde=False, label='Defect #4')

plt.legend()

plt.title('Mask Area Histogram: Defect #4', fontsize=15)



plt.tight_layout()

plt.show()
plt.figure(figsize=(15,4))



plt.subplot(111)

sns.kdeplot(masks_df[masks_df['Label'] == '1']['Area'].values, label='Defict #1')

sns.kdeplot(masks_df[masks_df['Label'] == '2']['Area'].values, label='Defict #2')

sns.kdeplot(masks_df[masks_df['Label'] == '3']['Area'].values, label='Defict #3')

sns.kdeplot(masks_df[masks_df['Label'] == '4']['Area'].values, label='Defict #4')

plt.legend()



plt.title('Mask Area KDE plot', fontsize=15)



def plot_image_grid(df, n_images = 5):

    

    fig, axs = plt.subplots(n_images, 2, figsize=(20, 10))

    

    for i in range(n_images):

        image_id = np.random.randint(0,len(df),1)[0]



        image = Image.open(TRAIN_PATH + df.iloc[image_id]['Image'])

        mask = rle2maskResize(df.iloc[image_id]['EncodedPixels'])

        

        defect = df.iloc[image_id]['Label']



        axs[i,0].imshow(image)

        axs[i,0].axis('off')

        axs[i,0].set_title('Original Image')



        axs[i, 1].imshow(image)

        axs[i, 1].imshow(mask, alpha = 0.5, cmap = "Reds")

        axs[i, 1].axis('off')

        axs[i, 1].set_title('Mask with defect #{}'.format(defect))



    plt.suptitle('Images with defect masks')
# filter the dataframe, so we have only images with very large masks

large_masks_df = masks_df[masks_df['Area'] > 200000]
# plot a grid of images with large masks

plot_image_grid(large_masks_df, n_images = 5)