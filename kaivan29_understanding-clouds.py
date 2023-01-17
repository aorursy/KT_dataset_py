import os

import gc

import sys

import time

import json

import glob

import random

from pathlib import Path

import pandas as pd



from PIL import Image

import cv2

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from imgaug import augmenters as iaa



import itertools

from tqdm import tqdm
sample_submission = pd.read_csv("../input/understanding_cloud_organization/sample_submission.csv")

train = pd.read_csv("../input/understanding_cloud_organization/train.csv")

train = train.dropna()

train.head()
category_list = ["Fish","Flower","Gravel","Sugar"]
train_dict = {}

train_class_dict = {}

for idx, row in train.iterrows():

    image_filename = row.Image_Label.split("_")[0]

    class_name = row.Image_Label.split("_")[1]

    class_id = category_list.index(class_name)

    if train_dict.get(image_filename):

        train_dict[image_filename].append(row.EncodedPixels)

        train_class_dict[image_filename].append(class_id)

    else:

        train_dict[image_filename] = [row.EncodedPixels]

        train_class_dict[image_filename] = [class_id]
df = pd.DataFrame(columns=["image_id","EncodedPixels","CategoryId","Width","Height"])

for key, value in train_dict.items():

    img = Image.open("../input/understanding_cloud_organization/train_images/{}".format(key))

    width, height = img.width, img.height

    df = df.append({"image_id": key, "EncodedPixels": value, "CategoryId": train_class_dict[key], "Width": width, "Height": height},ignore_index=True)
df.head()
DATA_DIR = Path('../kaggle/input/')

ROOT_DIR = "../../working"



NUM_CATS = len(category_list)

IMAGE_SIZE = 512
!git clone https://www.github.com/matterport/Mask_RCNN.git

os.chdir('Mask_RCNN')



!rm -rf .git # to prevent an error when the kernel is committed

!rm -rf images assets # to prevent displaying images at the bottom of a kernel
sys.path.append(ROOT_DIR+'/Mask_RCNN')

from mrcnn.config import Config



from mrcnn import utils

import mrcnn.model as modellib

from mrcnn import visualize

from mrcnn.model import log
!wget --quiet https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5

!ls -lh mask_rcnn_coco.h5



COCO_WEIGHTS_PATH = 'mask_rcnn_coco.h5'
class CloudConfig(Config):

    NAME = "cloud"

    NUM_CLASSES = NUM_CATS + 1 # +1 for the background class

    

    GPU_COUNT = 1

    IMAGES_PER_GPU = 4 #That is the maximum with the memory available on kernels

    

    BACKBONE = 'resnet50'

    

    IMAGE_MIN_DIM = IMAGE_SIZE

    IMAGE_MAX_DIM = IMAGE_SIZE    

    IMAGE_RESIZE_MODE = 'none'

    

    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)

    

    # STEPS_PER_EPOCH should be the number of instances 

    # divided by (GPU_COUNT*IMAGES_PER_GPU), and so should VALIDATION_STEPS;

    # however, due to the time limit, I set them so that this kernel can be run in 9 hours

    STEPS_PER_EPOCH = 4500

    VALIDATION_STEPS = 500

    

config = CloudConfig()

config.display()
def resize_image(image_path):

    img = cv2.imread(image_path)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)  

    return img
class CloudDataset(utils.Dataset):



    def __init__(self, df):

        super().__init__(self)

        

        # Add classes

        for i, name in enumerate(category_list):

            self.add_class("cloud", i+1, name)

        

        # Add images 

        for i, row in df.iterrows():

            self.add_image("cloud", 

                           image_id=row.name, 

                           path='../../input/understanding_cloud_organization/train_images/'+str(row.image_id), 

                           labels=row['CategoryId'],

                           annotations=row['EncodedPixels'], 

                           height=row['Height'], width=row['Width'])



    def image_reference(self, image_id):

        info = self.image_info[image_id]

        return info['path'], [category_list[int(x)] for x in info['labels']]

    

    def load_image(self, image_id):

        return resize_image(self.image_info[image_id]['path'])



    def load_mask(self, image_id):

        info = self.image_info[image_id]

                

        mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE, len(info['annotations'])), dtype=np.uint8)

        labels = []

        

        for m, (annotation, label) in enumerate(zip(info['annotations'], info['labels'])):

            sub_mask = np.full(info['height']*info['width'], 0, dtype=np.uint8)

            annotation = [int(x) for x in annotation.split(' ')]

            

            for i, start_pixel in enumerate(annotation[::2]):

                sub_mask[start_pixel: start_pixel+annotation[2*i+1]] = 1



            sub_mask = sub_mask.reshape((info['height'], info['width']), order='F')

            sub_mask = cv2.resize(sub_mask, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)

            

            mask[:, :, m] = sub_mask

            labels.append(int(label)+1)

            

        return mask, np.array(labels)
training_percentage = 0.9



training_set_size = int(training_percentage*len(df))

validation_set_size = int((1-training_percentage)*len(df))



train_dataset = CloudDataset(df[:training_set_size])

train_dataset.prepare()



valid_dataset = CloudDataset(df[training_set_size:training_set_size+validation_set_size])

valid_dataset.prepare()



for i in range(5):

    image_id = random.choice(train_dataset.image_ids)

    print(train_dataset.image_reference(image_id))

    

    image = train_dataset.load_image(image_id)

    mask, class_ids = train_dataset.load_mask(image_id)

    visualize.display_top_masks(image, mask, class_ids, train_dataset.class_names, limit=4)