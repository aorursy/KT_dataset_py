import os

import pandas as pd

import cv2

import numpy as np

from PIL import Image

from matplotlib import pyplot as plt

os.listdir('../input')
print('We have {} files in dataset'.format(len(os.listdir('../input/understanding_cloud_organization/train_images/'))))
# Reading the training dataset

df = pd.read_csv('../input/understanding_cloud_organization/train.csv')

df.tail()
# Split the labels from the Images ID

new = df.Image_Label.str.split('_', expand=True).rename(columns={0:'id',1:'labels'})

df['id']=new['id']

df['labels']=new['labels']

df.head()
# All individual labels

labels_counts = df.labels.value_counts()

labels_counts
print('We have {} NaN classes'.format(df.EncodedPixels.isna().sum()))
# Plotting the nan class occurance

value_count = df[df.EncodedPixels.isna()]['labels'].value_counts()

value_count.plot.bar()
# Plotting the classes occurances

non_nan_labels = labels_counts - value_count

non_nan_labels.plot.bar()
def np_resize(img, input_shape):

    """

    Reshape a numpy array, which is input_shape=(height, width), 

    as opposed to input_shape=(width, height) for cv2

    """

    height, width = input_shape

    return cv2.resize(img, (width, height))

    

def mask2rle(img):

    '''

    img: numpy array, 1 - mask, 0 - background

    Returns run length as string formated

    '''

    pixels= img.T.flatten()

    pixels = np.concatenate([[0], pixels, [0]])

    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    runs[1::2] -= runs[::2]

    return ' '.join(str(x) for x in runs)



def rle2mask(rle, input_shape):

    width, height = input_shape[:2]

    

    mask= np.zeros( width*height ).astype(np.uint8)

    

    array = np.asarray([int(x) for x in rle.split()])

    starts = array[0::2]

    lengths = array[1::2]



    current_position = 0

    for index, start in enumerate(starts):

        mask[int(start):int(start+lengths[index])] = 1

        current_position += lengths[index]

        

    return mask.reshape(height, width).T



def build_masks(rles, input_shape, reshape=None):

    depth = len(rles)

    if reshape is None:

        masks = np.zeros((*input_shape, depth))

    else:

        masks = np.zeros((*reshape, depth))

    

    for i, rle in enumerate(rles):

        if type(rle) is str:

            if reshape is None:

                masks[:, :, i] = rle2mask(rle, input_shape)

            else:

                mask = rle2mask(rle, input_shape)

                reshaped_mask = np_resize(mask, reshape)

                masks[:, :, i] = reshaped_mask

    

    return masks



def build_rles(masks, reshape=None):

    width, height, depth = masks.shape

    

    rles = []

    

    for i in range(depth):

        mask = masks[:, :, i]

        

        if reshape:

            mask = mask.astype(np.float32)

            mask = np_resize(mask, reshape).astype(np.int64)

        

        rle = mask2rle(mask)

        rles.append(rle)

        

    return rles
os.listdir('../input/understanding_cloud_organization/train_images/')[:10]
sample_filename = '8db703a.jpg'

sample_image_df = df[df['id'] == sample_filename]

sample_path = f"../input/understanding_cloud_organization/train_images/{sample_image_df['id'].iloc[0]}"

sample_img = cv2.imread(sample_path)

sample_rles = sample_image_df['EncodedPixels'].values

sample_masks = build_masks(sample_rles, input_shape=(1400, 2100))



fig, axs = plt.subplots(5, figsize=(12, 12))

axs[0].imshow(sample_img, cmap='gray')

axs[0].axis('off')



for i in range(4):

    axs[i+1].imshow(sample_masks[:, :, i])

#     axs[i+1].axis('off')
maskid=2

ymin = sample_masks[:,:,maskid].argmax(axis=1).argmax()

xmin = sample_masks[:,:,maskid].argmax(axis=0).argmax()

ymax = sample_masks[ymin:,xmin:,maskid].argmin(axis=1).argmin()+ymin

xmax = sample_masks[ymin:,xmin:,maskid].argmin(axis=0).argmin()+xmin



print(xmin, ymin, xmax, ymax)
sample_masks[ymin:,:,maskid].argmin(axis=1).shape
# (sample_img[ymin:ymax, xmin:xmax], cmap='gray')

# cv2.rectangle(sample_img, (xmin, ymin), (xmax, ymax), (0,255,0), 5)

plt.imshow(sample_img[ymin:ymax,xmin:xmax], cmap='gray')

plt.axis('off')