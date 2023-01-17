import os
import sys
import random
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

from tqdm import tqdm_notebook, tnrange
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from skimage.feature import canny
from skimage.filters import sobel,threshold_otsu, threshold_niblack,threshold_sauvola
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries
from scipy import signal

import cv2
from PIL import Image
import pdb
from tqdm import tqdm
import seaborn as sns
import os 
from glob import glob

import warnings
warnings.filterwarnings("ignore")
INPUT_PATH = '../input'
DATA_PATH = INPUT_PATH
TRAIN_DATA = os.path.join(DATA_PATH, "train_v2")
TRAIN_MASKS_DATA = os.path.join(DATA_PATH, "train_v2/masks")
TEST_DATA = os.path.join(DATA_PATH, "test_v2")
df = pd.read_csv(DATA_PATH+'/train_ship_segmentations_v2.csv')
path_train = '../input/train/'
path_test = '../input/test/'
train_ids = df.ImageId.values
df = df.set_index('ImageId')

## Gets full path of a image given the image name and image type(test or train)
def get_filename(image_id, image_type):
    check_dir = False
    if "Train" == image_type:
        data_path = TRAIN_DATA
    elif "mask" in image_type:
        data_path = TRAIN_MASKS_DATA
    elif "Test" in image_type:
        data_path = TEST_DATA
    else:
        raise Exception("Image type '%s' is not recognized" % image_type)

    if check_dir and not os.path.exists(data_path):
        os.makedirs(data_path)

    return os.path.join(data_path, "{}".format(image_id))

def get_image_data(image_id, image_type, **kwargs):
    img = _get_image_data_opencv(image_id, image_type, **kwargs)
    img = img.astype('uint8')
    return img

## Function to read image and return it 
def _get_image_data_opencv(image_id, image_type, **kwargs):
    fname = get_filename(image_id, image_type)
    img = cv2.imread(fname)
    assert img is not None, "Failed to read image : %s, %s" % (image_id, image_type)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

# ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
def rle_decode(mask_rle, shape=(768, 768)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction

# https://github.com/ternaus/TernausNet/blob/master/Example.ipynb
def mask_overlay(image, mask):
    """
    Helper function to visualize mask
    """
    mask = mask.astype(np.uint8)
    weighted_sum = cv2.addWeighted(mask, 0.75, image, 0.5, 0.)
    img = image.copy()
    ind = mask[:, :, 1] > 0    
    img[ind] = weighted_sum[ind]    
    return img
import traceback
import sys

## This function neatly displays the images in grid , we have option of showing masked / unmasked images.
def show_image(df,train_ids,show_masked = True , show_unmasked = True,plot_no_ship_images=False):
    ## We want to view 32 images in 4 rows
    nImg = 32  #no. of images that you want to display
    np.random.seed(42)
    if df.index.name == 'ImageId':
        df = df.reset_index()
    if df.index.name != 'ImageId':
        df = df.set_index('ImageId')

    _train_ids = list(train_ids)
    np.random.shuffle(_train_ids)
    tile_size = (256, 256)
    ## images per row
    n = 8
    alpha = 0.3

    ## Number of rows
    m = int(np.ceil(nImg * 1.0 / n))
    complete_image = np.zeros((m*(tile_size[0]+2), n*(tile_size[1]+2), 3), dtype=np.uint8)
    complete_image_masked = np.zeros((m*(tile_size[0]+2), n*(tile_size[1]+2), 3), dtype=np.uint8)

    counter = 0
    for i in range(m):
        ## For each row set up the row template for images
        ys = i*(tile_size[1] + 2)
        ye = ys + tile_size[1]
        j = 0
        while j < n:
            ## Now for each of images , load the image untill the we get 32 images
            counter += 1
            all_masks = np.zeros((768, 768))
            xs = j*(tile_size[0] + 2)
            xe = xs + tile_size[0]
            image_id = _train_ids[counter]
            ## For initial image exploration we would like to not have images with no ship , this can be toggle via the plot_no_ship_images option.
            if str(df.loc[image_id,'EncodedPixels'])==str(np.nan):
                if plot_no_ship_images:
                    j +=1
                else:    
                    continue
            else:
                j += 1
            img = get_image_data(image_id, 'Train')

            try:
                ## Depending on what type of images we want to see , compute the image matrix
                
                if show_unmasked:
                    img_resized = cv2.resize(img, dsize=tile_size)
                    img_with_text = cv2.putText(img_resized, image_id, (5,img.shape[0] - 5), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), thickness=2)
                    complete_image[ys:ye, xs:xe, :] = img_with_text[:,:,:]
                    
                if show_masked:
                    img_masks = df.loc[image_id,'EncodedPixels'].tolist()
                    for mask in img_masks:
                        all_masks += rle_decode(mask)
                    all_masks = np.expand_dims(all_masks,axis=2)
                    all_masks = np.repeat(all_masks,3,axis=2).astype('uint8')*255

                    img_masked = mask_overlay(img, all_masks)        
                    img_masked = cv2.resize(img_masked, dsize=tile_size)

                    img_masked = cv2.putText(img_masked, image_id, (5,img.shape[0] - 5), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), thickness=2)
                    complete_image_masked[ys:ye, xs:xe, :] = img_masked[:,:,:]

            except Exception as e:
                all_masks = rle_decode(df.loc[image_id,'EncodedPixels'])
                all_masks = np.expand_dims(all_masks,axis=2)*255
                all_masks = np.repeat(all_masks,3,axis=2).astype('uint8')

                img_masked = mask_overlay(img, all_masks)        

                img = cv2.resize(img, dsize=tile_size)
                img_masked = cv2.resize(img_masked, dsize=tile_size)

                img = cv2.putText(img, image_id, (5,img.shape[0] - 5), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), thickness=2)
                complete_image[ys:ye, xs:xe, :] = img[:,:,:]

                img_masked = cv2.putText(img_masked, image_id, (5,img.shape[0] - 5), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), thickness=2)
                complete_image_masked[ys:ye, xs:xe, :] = img_masked[:,:,:]
                
    ## Now plot images based on the options
    if show_unmasked:
        m = complete_image.shape[0] / (tile_size[0] + 2)
        k = 8
        n = int(np.ceil(m / k))
        for i in range(n):
            plt.figure(figsize=(20, 20))
            ys = i*(tile_size[0] + 2)*k
            ye = min((i+1)*(tile_size[0] + 2)*k, complete_image.shape[0])
            plt.imshow(complete_image[ys:ye,:,:],cmap='seismic')
            plt.title("Training dataset")
            
    if show_masked:
        m = complete_image.shape[0] / (tile_size[0] + 2)
        k = 8
        n = int(np.ceil(m / k))
        for i in range(n):
            plt.figure(figsize=(20, 20))
            ys = i*(tile_size[0] + 2)*k
            ye = min((i+1)*(tile_size[0] + 2)*k, complete_image.shape[0])
            plt.imshow(complete_image_masked[ys:ye,:,:])
            plt.title("Training dataset: Lighter Color depicts ship")

##Lets quickly test the function we just wrote            
show_image(df,train_ids)        

df = df.reset_index()
df['ship_count'] = df.groupby('ImageId')['ImageId'].transform('count')
df.loc[df['EncodedPixels'].isnull().values,'ship_count'] = 0  #see infocusp's comment
sns.set_style("white")
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
sns.distplot(df['ship_count'],kde=False)
plt.title('Ship Count Distribution in Train Set')

print(df['ship_count'].describe())
images_with_noships = df[df["ship_count"] == 0].ImageId.values
show_image(df,images_with_noships,show_masked=False,plot_no_ship_images=True)        
images_with_1_5 = df[df["ship_count"].between(1,5)].ImageId.values
show_image(df,images_with_1_5,show_unmasked=False,show_masked=True,plot_no_ship_images=True)        
images_with_5_10 = df[df["ship_count"].between(5,10)].ImageId.values
show_image(df,images_with_5_10,show_unmasked=False,show_masked=True,plot_no_ship_images=True)        
images_with_greater_10 = df[df["ship_count"].between(10,16)].ImageId.values
show_image(df,images_with_greater_10,show_unmasked=False,show_masked=True,plot_no_ship_images=True)        
