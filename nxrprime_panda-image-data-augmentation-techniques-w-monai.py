%reload_ext autoreload

%autoreload 2

%matplotlib inline

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import openslide

import os

!pip install monai

import torch
train = pd.read_csv('../input/prostate-cancer-grade-assessment/train.csv')

gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

gpu
def show_images(df, read_region=(1780,1950)):

    data = df

    f, ax = plt.subplots(3,3, figsize=(16,18))

    for i,data_row in enumerate(data.iterrows()):

        image = str(data_row[1][0])+'.tiff'

        image_path = os.path.join('../input/prostate-cancer-grade-assessment',"train_images",image)

        image = openslide.OpenSlide(image_path)

        spacing = 1 / (float(image.properties['tiff.XResolution']) / 10000)

        patch = image.read_region(read_region, 0, (256, 256))

        ax[i//3, i%3].imshow(patch) 

        image.close()       

        ax[i//3, i%3].axis('off')

        ax[i//3, i%3].set_title(f'ID: {data_row[1][0]}\nSource: {data_row[1][1]} ISUP: {data_row[1][2]} Gleason: {data_row[1][3]}')



    plt.show()

images = [

    '059cbf902c5e42972587c8d17d49efed', '06a0cbd8fd6320ef1aa6f19342af2e68', '06eda4a6faca84e84a781fee2d5f47e1',

    '037504061b9fba71ef6e24c48c6df44d', '035b1edd3d1aeeffc77ce5d248a01a53', '046b35ae95374bfb48cdca8d7c83233f',

    '074c3e01525681a275a42282cd21cbde', '05abe25c883d508ecc15b6e857e59f32', '05f4e9415af9fdabc19109c980daf5ad']   

data_sample = train.loc[train.image_id.isin(images)]

show_images(data_sample)
from monai.transforms import *



def show_images_affine(df, read_region=(1780,1950)):

    data = df

    f, ax = plt.subplots(3,3, figsize=(16,18))

    for i,data_row in enumerate(data.iterrows()):

        image = str(data_row[1][0])+'.tiff'

        image_path = os.path.join('../input/prostate-cancer-grade-assessment',"train_images",image)

        image = openslide.OpenSlide(image_path)

        spacing = 1 / (float(image.properties['tiff.XResolution']) / 10000)

        patch = image.read_region(read_region, 0, (256, 256))

        patch = np.array(patch)

        # MONAI transforms always take channel-first data: [channel x H x W]

        im_data = np.moveaxis(patch, -1, 0)  # make them channel first

        # create an Affine transform

        affine = Affine(rotate_params=np.pi/4, scale_params=(1.2, 1.2), translate_params=(200, 40), 

                padding_mode='zeros', device=torch.device('cuda:0'))

        # convert both image and segmentation using different interpolation mode

        new_img = affine(im_data, (256, 256), mode='bilinear')

        

        ax[i//3, i%3].imshow(np.moveaxis(new_img.astype(int), 0, -1)) 

        image.close()       

        ax[i//3, i%3].axis('off')

        ax[i//3, i%3].set_title(f'Affine Transformed\n Gleason: {data_row[1][3]}')

    plt.show()

    

images = [

    '059cbf902c5e42972587c8d17d49efed', '06a0cbd8fd6320ef1aa6f19342af2e68', '06eda4a6faca84e84a781fee2d5f47e1',

    '037504061b9fba71ef6e24c48c6df44d', '035b1edd3d1aeeffc77ce5d248a01a53', '046b35ae95374bfb48cdca8d7c83233f',

    '074c3e01525681a275a42282cd21cbde', '05abe25c883d508ecc15b6e857e59f32', '05f4e9415af9fdabc19109c980daf5ad']   

data_sample = train.loc[train.image_id.isin(images)]

show_images_affine(data_sample)
def show_images_elastic2d(df, read_region=(1780,1950)):

    data = df

    f, ax = plt.subplots(3,3, figsize=(16,18))

    for i,data_row in enumerate(data.iterrows()):

        image = str(data_row[1][0])+'.tiff'

        image_path = os.path.join('../input/prostate-cancer-grade-assessment',"train_images",image)

        image = openslide.OpenSlide(image_path)

        spacing = 1 / (float(image.properties['tiff.XResolution']) / 10000)

        patch = image.read_region(read_region, 0, (256, 256))

        patch = np.array(patch)

        im_data = np.moveaxis(patch, -1, 0)  # make them channel first

        # create an elastic transform

        deform = Rand2DElastic(prob=1.0, spacing=(30, 30), magnitude_range=(5, 6),

                       rotate_range=(np.pi/4,), scale_range=(0.2, 0.2), translate_range=(100, 100), 

                       padding_mode='zeros', device=torch.device('cuda:0'))

        new_img = deform(im_data, (256, 256), mode='nearest')

        

        ax[i//3, i%3].imshow(np.moveaxis(new_img.astype(int), 0, -1)) 

        image.close()       

        ax[i//3, i%3].axis('off')

        ax[i//3, i%3].set_title(f'Deformed Transformed\n Gleason: {data_row[1][3]}')

    plt.show()

    

images = [

    '059cbf902c5e42972587c8d17d49efed', '06a0cbd8fd6320ef1aa6f19342af2e68', '06eda4a6faca84e84a781fee2d5f47e1',

    '037504061b9fba71ef6e24c48c6df44d', '035b1edd3d1aeeffc77ce5d248a01a53', '046b35ae95374bfb48cdca8d7c83233f',

    '074c3e01525681a275a42282cd21cbde', '05abe25c883d508ecc15b6e857e59f32', '05f4e9415af9fdabc19109c980daf5ad']   

data_sample = train.loc[train.image_id.isin(images)]

show_images_elastic2d(data_sample)
def show_images_rotate(df, read_region=(1780,1950)):

    data = df

    f, ax = plt.subplots(3,3, figsize=(16,18))

    for i,data_row in enumerate(data.iterrows()):

        image = str(data_row[1][0])+'.tiff'

        image_path = os.path.join('../input/prostate-cancer-grade-assessment',"train_images",image)

        image = openslide.OpenSlide(image_path)

        spacing = 1 / (float(image.properties['tiff.XResolution']) / 10000)

        patch = image.read_region(read_region, 0, (256, 256))

        patch = np.array(patch)

        im_data = np.moveaxis(patch, -1, 0)  # make them channel first

        rotater = Flip(spatial_axis=1)

        new_img = rotater(im_data)

        

        ax[i//3, i%3].imshow(np.moveaxis(new_img.astype(int), 0, -1)) 

        image.close()       

        ax[i//3, i%3].axis('off')

        ax[i//3, i%3].set_title(f'Rotated Transformed\n Gleason: {data_row[1][3]}')

    plt.show()

    

images = [

    '059cbf902c5e42972587c8d17d49efed', '06a0cbd8fd6320ef1aa6f19342af2e68', '06eda4a6faca84e84a781fee2d5f47e1',

    '037504061b9fba71ef6e24c48c6df44d', '035b1edd3d1aeeffc77ce5d248a01a53', '046b35ae95374bfb48cdca8d7c83233f',

    '074c3e01525681a275a42282cd21cbde', '05abe25c883d508ecc15b6e857e59f32', '05f4e9415af9fdabc19109c980daf5ad']   

data_sample = train.loc[train.image_id.isin(images)]

show_images_rotate(data_sample)
def show_images_rotate(df, read_region=(1780,1950)):

    data = df

    f, ax = plt.subplots(3,3, figsize=(16,18))

    for i,data_row in enumerate(data.iterrows()):

        image = str(data_row[1][0])+'.tiff'

        image_path = os.path.join('../input/prostate-cancer-grade-assessment',"train_images",image)

        image = openslide.OpenSlide(image_path)

        spacing = 1 / (float(image.properties['tiff.XResolution']) / 10000)

        patch = image.read_region(read_region, 0, (256, 256))

        patch = np.array(patch)

        im_data = np.moveaxis(patch, -1, 0)  # make them channel first

        rotater = SpatialPad(spatial_size=(300, 300), mode='mean')

        new_img = rotater(im_data)

        

        ax[i//3, i%3].imshow(np.moveaxis(new_img.astype(int), 0, -1)) 

        image.close()       

        ax[i//3, i%3].axis('off')

        ax[i//3, i%3].set_title(f'Rotated Transformed\n Gleason: {data_row[1][3]}')

    plt.show()

    

images = [

    '059cbf902c5e42972587c8d17d49efed', '06a0cbd8fd6320ef1aa6f19342af2e68', '06eda4a6faca84e84a781fee2d5f47e1',

    '037504061b9fba71ef6e24c48c6df44d', '035b1edd3d1aeeffc77ce5d248a01a53', '046b35ae95374bfb48cdca8d7c83233f',

    '074c3e01525681a275a42282cd21cbde', '05abe25c883d508ecc15b6e857e59f32', '05f4e9415af9fdabc19109c980daf5ad']   

data_sample = train.loc[train.image_id.isin(images)]

show_images_rotate(data_sample)
import cv2

 

image = cv2.imread('../input/panda-resize-and-save-train-data/0005f7aaab2800f6170c399693a96917.png')

image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

image = cv2.resize(image, (256, 256))

image=cv2.addWeighted ( image,4, cv2.GaussianBlur( image , (0,0) , 256/10) ,-4 ,128) # the trick is to add this line

plt.imshow(image)

plt.title('Ben Graham Method\n Transformed image')

plt.show()
