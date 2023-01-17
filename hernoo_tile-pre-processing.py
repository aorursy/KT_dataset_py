#inspired by https://www.kaggle.com/rftexas/better-image-tiles-removing-white-spaces
#depedencies

import os

import cv2

import PIL

import random

import openslide

import skimage.io

import matplotlib

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from IPython.display import Image, display
sample_number='NoLimit'



if sample_number=='NoLimit':

    train_df = pd.read_csv('../input/prostate-cancer-grade-assessment/train.csv').reset_index(drop=True)

else:

    train_df = pd.read_csv('../input/prostate-cancer-grade-assessment/train.csv').reset_index(drop=True).sample(n=sample_number)



    #train_df = pd.read_csv('../input/prostate-cancer-grade-assessment/train.csv').sample(n=100, random_state=0).reset_index(drop=True)



images = list(train_df['image_id'])

len_image=len(images)

labels = list(train_df['isup_grade'])

data_dir = '../input/prostate-cancer-grade-assessment/train_images/'
#Functions

## to get the % of different colors 

def compute_statistics(image):

    """

    Args:

        image                  numpy.array   multi-dimensional array of the form WxHxC

    

    Returns:

        ratio_white_pixels     float         ratio of white pixels over total pixels in the image 

    """

    width, height = image.shape[0], image.shape[1]

    num_pixels = width * height

    

    num_white_pixels = 0

    

    summed_matrix = np.sum(image, axis=-1)

    # Note: A 3-channel white pixel has RGB (255, 255, 255)

    num_white_pixels = np.count_nonzero((summed_matrix > 620)) #avoid too white and/or too blank

    

    ratio_white_pixels = num_white_pixels / num_pixels

    

    green_concentration = np.mean(image[1])

    blue_concentration = np.mean(image[2])

    red_median = np.percentile(image[0],50)

    green_median = np.percentile(image[1],50)

    blue_median = np.percentile(image[2],50)

    return ratio_white_pixels, green_concentration, blue_concentration, red_median, green_median, blue_median



#selection of the k-best regions

def select_k_best_regions(regions, k=20):

    """

    Args:

        regions               list           list of 2-component tuples first component the region, 

                                             second component the ratio of white pixels

                                             

        k                     int            number of regions to select

    """

    red_penalty=0

    #regions = [x for x in regions if ((x[3] > 180 and x[4] > 180) and (((x[5]-red_penalty)>x[6]) or ((x[5]-red_penalty)>x[7])))] # x[3] is green concentration and 4 is blue 

    regions = [x for x in regions if (x[3] > 180 and x[4] > 180)] # x[3] is green concentration and 4 is blue 

    k_best_regions = sorted(regions, key=lambda tup: tup[2])[:k]

    return k_best_regions

#to retrieve from the top left pixel the full region



def get_k_best_regions(coordinates, image, window_size=512):

    regions = {}

    for i, tup in enumerate(coordinates):

        x, y = tup[0], tup[1]

        regions[i] = image[x : x+window_size, y : y+window_size, :]

    

    return regions



#the slider

def generate_patches(slide_path, window_size=200, stride=128, k=20):

    

    image = skimage.io.MultiImage(slide_path)[-2]

    image = np.array(image)

    

    max_width, max_height = image.shape[0], image.shape[1]

    regions_container = []

    i = 0

    

    while window_size + stride*i <= max_height:

        j = 0

        

        while window_size + stride*j <= max_width:            

            x_top_left_pixel = j * stride

            y_top_left_pixel = i * stride

            

            patch = image[

                x_top_left_pixel : x_top_left_pixel + window_size,

                y_top_left_pixel : y_top_left_pixel + window_size,

                :

            ]

            

            ratio_white_pixels, green_concentration, blue_concentration, red_median, green_median, blue_median = compute_statistics(patch)

            

            region_tuple = (x_top_left_pixel, y_top_left_pixel, ratio_white_pixels, green_concentration, blue_concentration,  red_median, green_median, blue_median)

            regions_container.append(region_tuple)

            #print(f' DEBUG : rmed :{red_median} gmed :{green_median} bmed : {blue_median}')

            j += 1

        

        i += 1

    

    k_best_region_coordinates = select_k_best_regions(regions_container, k=k)

    k_best_regions = get_k_best_regions(k_best_region_coordinates, image, window_size)

    

    return image, k_best_region_coordinates, k_best_regions





#showing results 

def display_images(regions, title):

    fig, ax = plt.subplots(5, 4, figsize=(15, 15))

    

    for i, region in regions.items():

        ax[i//4, i%4].imshow(region)

    

    fig.suptitle(title)

    

## glue to a unique picture

def glue_to_one_picture(image_patches, window_size=200, k=16):

    side = int(np.sqrt(k))

    image = np.zeros((side*window_size, side*window_size, 3), dtype=np.int16)

        

    for i, patch in image_patches.items():

        x = i // side

        y = i % side

        image[

            x * window_size : (x+1) * window_size,

            y * window_size : (y+1) * window_size,

            :

        ] = patch

    

    return image
#%%time

#ex_url = data_dir + images[13] + '.tiff'

#_, best_coordinates, best_regions = generate_patches(ex_url)

#display_images(best_regions, 'Window size: 200, stride: 128')
#### %%time

WINDOW_SIZE = 128

STRIDE = 105

K = 16

#fig, ax = plt.subplots(30, 2, figsize=(20, 25))

if os.path.exists('/kaggle/working/512x512x3'): 

    print('directory already created, skipping')

else:

    os.mkdir('/kaggle/working/512x512x3')



for i, img in enumerate(images):

    url = data_dir + img + '.tiff'

    image, best_coordinates, best_regions = generate_patches(url, window_size=WINDOW_SIZE, stride=STRIDE, k=K)

    glued_image = glue_to_one_picture(best_regions, window_size=WINDOW_SIZE, k=K)

    

    #ax[i][0].imshow(image)

    #ax[i][0].set_title(f'{img} - Original - Label: {labels[i]}')

    

    #ax[i][1].imshow(glued_image)

    #ax[i][1].set_title(f'{img} - Glued - Label: {labels[i]}')

    #glued_image=glued_image/255 #normalisation 

    print(f'Image #{i} processed out of {len_image}')

    cv2.imwrite(f"/kaggle/working/512x512x3/{img}.png", glued_image)



#fig.suptitle('From biopsy to glued patches')

#img=images[13]

#url = data_dir + img + '.tiff'

#image, best_coordinates, best_regions = generate_patches(url, window_size=WINDOW_SIZE, stride=STRIDE, k=K)

#import matplotlib.pyplot as plt

#import matplotlib.image as mpimg

#img=mpimg.imread(f"/kaggle/working/testoutput/{img}.png")

#imgplot = plt.imshow(img)

#plt.show()
#bb=best_regions[0]

#r,g,b=bb[:,:,0],bb[:,:,1],bb[:,:,2]

#plt.imshow(bb)

#print(f'{np.mean(r)},{np.mean(g)},{np.mean(b)}')

#print(f'{np.percentile(r,80)},{np.percentile(g,80)},{np.percentile(b,80)}')



#bb=best_regions[8]

#r,g,b=bb[:,:,0],bb[:,:,1],bb[:,:,2]

#plt.imshow(bb)

#print(f'{np.mean(r)},{np.mean(g)},{np.mean(b)}')



#print(f'{np.median(r)},{np.median(g)},{np.median(b)}')

#bb=best_regions[9]

#r,g,b=bb[:,:,0],bb[:,:,1],bb[:,:,2]

#plt.imshow(bb)

#print(f'{np.mean(r)},{np.mean(g)},{np.mean(b)}')

#import imageio

#imageio.imread("./512x512x3/5b5d7aa9b4ded22e5c6d59eaad75684f.png")

#plt.imshow(imageio.imread("./512x512x3/5b5d7aa9b4ded22e5c6d59eaad75684f.png"))