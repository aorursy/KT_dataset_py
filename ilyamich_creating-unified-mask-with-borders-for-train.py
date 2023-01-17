import os
from tqdm import tqdm
import numpy as np
import random
import matplotlib as plt

from skimage import feature
from skimage.io import imread, imshow
from skimage.transform import SimilarityTransform, warp

%matplotlib inline
TRAIN_PATH = '../input/stage1_train/'

train_ids = next(os.walk(TRAIN_PATH))[1]
def create_unified_mask(image_path, mask_shape):
    
    final_mask = np.zeros(mask_shape)
    
    masks = next(os.walk(TRAIN_PATH + image_path + '/masks/'))[2]
        
    for mask in masks:
        m = imread(TRAIN_PATH + image_path + '/masks/' + mask)        
        final_mask = np.maximum(final_mask, m)
    
    return final_mask
image_path = train_ids[random.randint(0, len(train_ids))]
img = imread(TRAIN_PATH + image_path + '/images/' + image_path + '.png')
unified_mask = create_unified_mask(image_path, img.shape[0:2])

imshow(unified_mask)
def create_unified_mask_with_borders(image_path, mask_shape):
    
    final_mask = np.zeros(mask_shape)
    
    masks = next(os.walk(TRAIN_PATH + image_path + '/masks/'))[2]
        
    for mask in masks:
        m = imread(TRAIN_PATH + image_path + '/masks/' + mask) / 255
        
        image_left = warp(m, SimilarityTransform(translation=(2, 0)))
        image_right = warp(m, SimilarityTransform(translation=(-2, 0)))
        image_up = warp(m, SimilarityTransform(translation=(0, 2)))
        image_down = warp(m, SimilarityTransform(translation=(0, -2)))
        
        border = final_mask + image_left == 2
        border = np.logical_or(border, final_mask + image_right == 2)
        border = np.logical_or(border, final_mask + image_up == 2)
        border = np.logical_or(border, final_mask + image_down == 2)
        
        final_mask = np.maximum(final_mask, m)
        final_mask[border] = 0
    
    return final_mask
unified_mask_with_border = create_unified_mask_with_borders(image_path, img.shape[0:2])

imshow(unified_mask_with_border)