import os
import glob
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import SimpleITK as sitk
import skimage.io as io

import tensorflow as tf
img_files_1 = sorted(glob.glob('/kaggle/input/luna16p1/subset0/subset0/*.mhd'))
img_files_2 = sorted(glob.glob('/kaggle/input/luna16p1/subset1/subset1/*.mhd'))
img_files_3 = sorted(glob.glob('/kaggle/input/luna16p1/subset2/subset2/*.mhd'))
img_files_4 = sorted(glob.glob('/kaggle/input/luna16p1/subset3/subset3/*.mhd'))

train_img_files = img_files_1 + img_files_2 + img_files_3
valid_img_files = img_files_4

print(len(train_img_files))
print(len(valid_img_files))
lung_mask_dir = '/kaggle/input/luna16p1/seg-lungs-LUNA16/seg-lungs-LUNA16/'

train_msk_files = []
valid_msk_files = []

for f in train_img_files:
    file_name = f.split('/')[-1]
    file_path = lung_mask_dir + file_name
    train_msk_files.append(file_path)

for f in valid_img_files:
    file_name = f.split('/')[-1]
    file_path = lung_mask_dir + file_name
    valid_msk_files.append(file_path)
def read(file):
    img = io.imread(file.numpy().decode('utf-8'), plugin='simpleitk')
    img = img.transpose(1, 2, 0)
    return img



def tf_read(img_file, msk_file):
    [img,] = tf.py_function(read, [img_file], [tf.int16])
    [msk,] = tf.py_function(read, [msk_file], [tf.int16])
    return img, msk
def get_patches(image, patch_vol=(64,64,64)):
    
    # Padding 
    
    image_x, image_y, image_z = image.shape
    patch_x, patch_y, patch_z = patch_vol

    x_pad = (patch_x - image_x % patch_x) % patch_x
    y_pad = (patch_y - image_y % patch_y) % patch_y
    z_pad = (patch_z - image_z % patch_z) % patch_z

    padded_image = np.pad(image, ((0,x_pad),(0,y_pad),(0,z_pad)), 'edge')

    
    # Patching 

    image_x, image_y, image_z = padded_image.shape

    image_patches = []

    i=0
    while i<image_x:
        j=0
        while j<image_y:
            k=0
            while k<image_z:
                patch_image = padded_image[i:i+patch_x, j:j+patch_y, k:k+patch_z]
                image_patches.append(patch_image)
                k+=patch_z
            j+=patch_y
        i+=patch_x

    return np.array(image_patches)



def tf_get_patches(img, msk):
    patch_vol = (64,64,64)
    [img,] = tf.py_function(get_patches, [img, patch_vol], [tf.int16])
    [msk,] = tf.py_function(get_patches, [msk, patch_vol], [tf.int16])
    return img, msk
dataset = tf.data.Dataset.from_tensor_slices((train_img_files, train_msk_files))
dataset = dataset.map(tf_read)
dataset = dataset.map(tf_get_patches)
for i, j in dataset:
    break
    
print(i.shape)
print(j.shape)

