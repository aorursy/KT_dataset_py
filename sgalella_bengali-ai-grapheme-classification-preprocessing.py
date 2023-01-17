# Import the different modules

import os

import cv2

import zipfile

import PIL.Image

import numpy as np

import pandas as pd

import tensorflow as tf

import matplotlib.pyplot as plt

from tqdm.notebook import tqdm

print(tf.__version__)
train = pd.read_csv("../input/bengaliai-cv19/train.csv")

test = pd.read_csv("../input/bengaliai-cv19/test.csv")

class_map = pd.read_csv("../input/bengaliai-cv19/class_map.csv")
print("Training set size: {}".format(train.shape))

train.head()
print(f"No. different graphemes: {len(train['grapheme_root'].unique())}")

print(f"No. different vowels: {len(train['vowel_diacritic'].unique())}")

print(f"No. different consonants: {len(train['consonant_diacritic'].unique())}")
print("Test set size: {}".format(test.shape))

test.head()
print("Class map size: {}".format(class_map.shape))

class_map.head()
HEIGHT = 137

WIDTH = 236



# Load one parquet file to observe the different images

FOLDER_DIRECTORY = '/kaggle/input/bengaliai-cv19/'

train = pd.read_parquet(os.path.join(FOLDER_DIRECTORY, 'train_image_data_0.parquet'))
def display_image_from_data(df, size=5, height=HEIGHT, width=WIDTH):

    """

    Displays grapheme images from data.

    

    Args:

        df (pandas.dataframe):

        size (int): Number of graphemes to display

        height(int): Height of the images (num rows)

        width(int): Width of the images (num cols)

    """

    fig, ax = plt.subplots(size, size, figsize=(12, 12))

    for i, idx in enumerate(df.index):

        image_id = df.iloc[i]['image_id']

        flattened_image = df.iloc[i].drop('image_id').values.astype(np.uint8)

        unpacked_image = PIL.Image.fromarray(flattened_image.reshape(137, 236))

        ax[i//size, i%size].imshow(unpacked_image)

        ax[i//size, i%size].set_title(image_id)

        ax[i//size, i%size].axis('on')

        

display_image_from_data(train.sample(25))
# Final size of images after preprocessing (size x size)

SIZE = 224



def bounding_box(image):

    """

    Defines the bounding box containing the grapheme

    

    Args:

        image (numpy.ndarray): grapheme image

    """

    rows = np.any(image, axis=1)

    cols = np.any(image, axis=0)

    rmin, rmax = np.where(rows)[0][[0, -1]]

    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax



def crop_resize(image, size=SIZE, pad=16):

    """

    Crops, pads and resize the grapheme

    

    Args:

        image (numpy.ndarray): grapheme image

        size (int): final size of the croped image

        pad (int): Amount of padding to the image

    """

    ymin, ymax, xmin, xmax = bounding_box(image[5:-5, 5:-5]>80)

    xmin = xmin - 13 if (xmin > 13) else 0

    ymin = ymin - 10 if (ymin > 10) else 0

    xmax = xmax + 13 if (xmax < WIDTH - 13) else WIDTH

    ymax = ymax + 10 if (ymax < HEIGHT - 10) else HEIGHT

    image = image[ymin:ymax, xmin:xmax]

    image[image < 28] = 0

    lx, ly = xmax-xmin, ymax-ymin

    l = max(lx, ly) + pad

    image = np.pad(image, [((l-ly)//2, ), ((l-lx)//2, )], mode='constant')

    return cv2.resize(image, (SIZE, SIZE))



num_images = 5

train_sample = train.sample(num_images)

fig, ax = plt.subplots(num_images, 2, figsize=(12, 12))



for idx in range(num_images):

    image_nocrop = 255 - train_sample.iloc[idx, 1:].values.reshape(HEIGHT, WIDTH).astype(np.uint8)

    image_crop = (image_nocrop*(255/image_nocrop.max())).astype(np.uint8)

    image_crop = crop_resize(image_crop)

    ax[idx, 0].imshow(image_nocrop)

    ax[idx, 0].set_title('Original Image')

    ax[idx, 0].axis('off')

    ax[idx, 1].imshow(image_crop)

    ax[idx, 1].set_title('Cropped + Resized Image')

    ax[idx, 1].axis('off')

    
# Save the train processed images in a zip

TRAIN_FILES = ['train_image_data_0.parquet','train_image_data_1.parquet','train_image_data_2.parquet','train_image_data_3.parquet']



with zipfile.ZipFile('train.zip','w') as train_out:

    for file in TRAIN_FILES:

        file_df = pd.read_parquet(os.path.join(FOLDER_DIRECTORY, file))

        data = 255 - file_df.iloc[:, 1:].values.reshape(-1, HEIGHT, WIDTH).astype(np.uint8)

        for idx in tqdm(range(len(file_df))):

            name = file_df.iloc[idx,0]

            image = (data[idx]*(255/data[idx].max())).astype(np.uint8)

            image = crop_resize(image)

            train_out.writestr(name + '.png', cv2.imencode('.png', image)[1])
# Save the test processed images in a zip

TEST_FILES = ['test_image_data_0.parquet','test_image_data_1.parquet','test_image_data_2.parquet','test_image_data_3.parquet']



with zipfile.ZipFile('test.zip','w') as test_out:

    for file in TEST_FILES:

        file_df = pd.read_parquet(os.path.join(FOLDER_DIRECTORY, file))

        data = 255 - file_df.iloc[:, 1:].values.reshape(-1, HEIGHT, WIDTH).astype(np.uint8)

        for idx in tqdm(range(len(file_df))):

            name = file_df.iloc[idx,0]

            image = (data[idx]*(255/data[idx].max())).astype(np.uint8)

            image = crop_resize(image)

            test_out.writestr(name + '.png', cv2.imencode('.png', image)[1])