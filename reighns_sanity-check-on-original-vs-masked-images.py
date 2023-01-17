import os



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import openslide

import matplotlib.pyplot as plt

from PIL import Image

import cv2

from tqdm.notebook import tqdm

import skimage.io

from skimage.transform import resize, rescale

import glob
train = pd.read_csv('/kaggle/input/prostate-cancer-grade-assessment/train.csv')

train_images_path = '/kaggle/input/prostate-cancer-grade-assessment/train_images/'

train_label_mask_path = '/kaggle/input/prostate-cancer-grade-assessment/train_label_masks/'

img_type = "*.tiff"
def sanity_tally(train_images_path, train_label_mask_path, img_type):

    total_img_list = [os.path.basename(img_name) for img_name

                      in glob.glob(os.path.join(train_images_path, img_type))]

    ## get the image_name

    total_img_list = [x[:-5] for x in total_img_list]

    

    mask_img_list  = [os.path.basename(img_name) for img_name 

                      in glob.glob(os.path.join(train_label_mask_path, img_type))]

    

    # note that the image name in train_label_mask will always be in this format: abcdefg_mask.tiff; therefore I needed to

    # remove the last 10 characters to tally with the images in train_images.

    mask_img_list  = [x[:-10] for x in mask_img_list]

    set_diff1      = set(total_img_list) - set(mask_img_list)

    set_diff2      = set(mask_img_list)  - set(total_img_list)

    

    if set(total_img_list)  == set(mask_img_list):

        print("Sanity Check Status: True")

    else:

        print("Sanity Check Status: Failed. \nThe elements in train_images_path but not in the train_label_mask_path is {} and the number is {}.\n\n\nThe elements in train_label_mask_path but not in train_images_path is {} and the number is {}".format(

                set_diff1, len(set_diff1), set_diff2, len(set_diff2)))

    

    return set_diff1, set_diff2
set_diff1, set_diff2 = sanity_tally(train_images_path,train_label_mask_path, img_type)
remove_images = list(set_diff1)

new_train = train[~train.image_id.isin(remove_images)]
new_train = new_train.reset_index(drop=True)
new_train
save_dir = "/kaggle/train/"

os.makedirs(save_dir, exist_ok=True)



train_images_path = '/kaggle/input/prostate-cancer-grade-assessment/train_images/'

train_label_mask_path = '/kaggle/input/prostate-cancer-grade-assessment/train_label_masks/'

img_type = "*.tiff"
for img_id in tqdm(new_train.image_id):

    load_path = train_images_path + img_id + '.tiff'

    save_path = save_dir + img_id + '.jpg'

    

    biopsy = skimage.io.MultiImage(load_path)

    img = cv2.resize(biopsy[-1], (512, 512))

    cv2.imwrite(save_path, img)
for img_id in tqdm(new_train.image_id):

    load_path = train_label_mask_path + img_id + '_mask' + '.tiff'

    save_path = save_dir + img_id + '.jpg'

    

    biopsy = skimage.io.MultiImage(load_path)

    img = cv2.resize(biopsy[-1], (512, 512))

    cv2.imwrite(save_path, img)
!tar -czf images.tar.gz ../train/*.png