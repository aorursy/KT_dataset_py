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
train_df = pd.read_csv("../input/understanding_cloud_organization/train.csv")

train_df = train_df.dropna()
train_df.head()
category_list = ["Fish","Flower","Gravel","Sugar"]
train_dict = {}

train_class_dict = {}

for idx, row in train_df.iterrows():

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

    visualize.display_top_masks(image, mask, class_ids, train_dataset.class_names, limit=5)
LR = 1e-4

EPOCHS = [3,9]



import warnings 

warnings.filterwarnings("ignore")
augmentation = iaa.Sequential([

    iaa.Fliplr(0.5),

    iaa.Flipud(0.5)

], random_order=True)
model = modellib.MaskRCNN(mode='training', config=config, model_dir=ROOT_DIR)



model.load_weights(COCO_WEIGHTS_PATH, by_name=True, exclude=[

    'mrcnn_class_logits', 'mrcnn_bbox_fc', 'mrcnn_bbox', 'mrcnn_mask'])
ls
%%time

model.train(train_dataset, valid_dataset,

            learning_rate=LR*2,

            epochs=EPOCHS[0],

            layers='heads',

            augmentation=None)



history = model.keras_model.history.history
%%time

model.train(train_dataset, valid_dataset,

            learning_rate=LR,

            epochs=EPOCHS[1],

            layers='all',

            augmentation=augmentation)



new_history = model.keras_model.history.history

for k in new_history: history[k] = history[k] + new_history[k]
epochs = range(EPOCHS[-1])



plt.figure(figsize=(18, 6))



plt.subplot(131)

plt.plot(epochs, history['loss'], label="train loss")

plt.plot(epochs, history['val_loss'], label="valid loss")

plt.legend()

plt.subplot(132)

plt.plot(epochs, history['mrcnn_class_loss'], label="train class loss")

plt.plot(epochs, history['val_mrcnn_class_loss'], label="valid class loss")

plt.legend()

plt.subplot(133)

plt.plot(epochs, history['mrcnn_mask_loss'], label="train mask loss")

plt.plot(epochs, history['val_mrcnn_mask_loss'], label="valid mask loss")

plt.legend()



plt.show()
best_epoch = np.argmin(history["val_loss"]) + 1

print("Best epoch: ", best_epoch)

print("Valid loss: ", history["val_loss"][best_epoch-1])
class InferenceConfig(CloudConfig):

    GPU_COUNT = 1

    IMAGES_PER_GPU = 1



inference_config = InferenceConfig()



model = modellib.MaskRCNN(mode='inference', 

                          config=inference_config,

                          model_dir=ROOT_DIR)
glob_list = glob.glob(f'../../working/cloud*/mask_rcnn_cloud_{best_epoch:04d}.h5')

model_path = glob_list[0] if glob_list else ''

model.load_weights(model_path, by_name=True)
# Fix overlapping masks

def refine_masks(masks, rois):

    areas = np.sum(masks.reshape(-1, masks.shape[-1]), axis=0)

    mask_index = np.argsort(areas)

    union_mask = np.zeros(masks.shape[:-1], dtype=bool)

    for m in mask_index:

        masks[:, :, m] = np.logical_and(masks[:, :, m], np.logical_not(union_mask))

        union_mask = np.logical_or(masks[:, :, m], union_mask)

    for m in range(masks.shape[-1]):

        mask_pos = np.where(masks[:, :, m]==True)

        if np.any(mask_pos):

            y1, x1 = np.min(mask_pos, axis=1)

            y2, x2 = np.max(mask_pos, axis=1)

            rois[m, :] = [y1, x1, y2, x2]

    return masks, rois
sample_df = pd.read_csv("../../input/understanding_cloud_organization/sample_submission.csv")

sample_df.head()
test_df = pd.DataFrame(columns=["image_id","EncodedPixels","CategoryId"])

for idx,row in sample_df.iterrows():

    image_filename = row.Image_Label.split("_")[0]

    test_df = test_df.append({"image_id": image_filename},ignore_index=True)

test_df = test_df.drop_duplicates()
test_df.head()
for i in range(8):

    image_id = test_df.sample()["image_id"].values[0]

    image_path = str('../../input/understanding_cloud_organization/test_images/'+image_id)

    print(image_path)

    img = cv2.imread(image_path)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    

    result = model.detect([resize_image(image_path)])

    r = result[0]

    

    if r['masks'].size > 0:

        masks = np.zeros((img.shape[0], img.shape[1], r['masks'].shape[-1]), dtype=np.uint8)

        for m in range(r['masks'].shape[-1]):

            masks[:, :, m] = cv2.resize(r['masks'][:, :, m].astype('uint8'), 

                                        (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

        

        y_scale = img.shape[0]/IMAGE_SIZE

        x_scale = img.shape[1]/IMAGE_SIZE

        rois = (r['rois'] * [y_scale, x_scale, y_scale, x_scale]).astype(int)

        

        masks, rois = refine_masks(masks, rois)

    else:

        masks, rois = r['masks'], r['rois']

        

    visualize.display_instances(img, rois, masks, r['class_ids'], 

                                ['bg']+category_list, r['scores'],

                                title=image_id, figsize=(12, 12))
def rle_encoding(x):

    dots = np.where(x.T.flatten() == 1)[0]

    run_lengths = []

    prev = -2

    for b in dots:

        if (b>prev+1): run_lengths.extend((b + 1, 0))

        run_lengths[-1] += 1

        prev = b

    return ' '.join([str(x) for x in run_lengths])
submission_df = sample_df.copy()

submission_df["EncodedPixels"] = ""

with tqdm(total=len(test_df)) as pbar:

    for i,row in test_df.iterrows():

        pbar.update(1)

        image_id = row["image_id"]

        image_path = str('../../input/understanding_cloud_organization/test_images/'+image_id)

        img = cv2.imread(image_path)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)



        result = model.detect([resize_image(image_path)])

        r = result[0]



        if r['masks'].size > 0:

            masks = np.zeros((img.shape[0], img.shape[1], r['masks'].shape[-1]), dtype=np.uint8)

            for m in range(r['masks'].shape[-1]):

                masks[:, :, m] = cv2.resize(r['masks'][:, :, m].astype('uint8'), 

                                            (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)



            y_scale = img.shape[0]/IMAGE_SIZE

            x_scale = img.shape[1]/IMAGE_SIZE

            rois = (r['rois'] * [y_scale, x_scale, y_scale, x_scale]).astype(int)

            masks, rois, class_ids = r['masks'], r['rois'], r['class_ids']



            #The following piece of code is creating rectangular masks from

            # the ROIs instead of using the masks drawn by the MaskRCNN.

            # It also removes any missing area from the imagery from the predicted masks.

            # Everything is added directly to the submission dataframe.

            rectangular_masks = []

            mask_dict = {"Fish":[],"Flower":[],"Gravel":[],"Sugar":[]}

            for roi, class_id in zip(rois, class_ids):

                rectangular_mask = np.zeros((512,512))

                rectangular_mask[roi[0]:roi[2], roi[1]:roi[3]] = 255

                img = cv2.resize(img, dsize=(512,512), interpolation = cv2.INTER_LINEAR)

                cropped_img = img[roi[0]:roi[2], roi[1]:roi[3]]

                

                kernel = np.ones((5,5),np.uint8)

                missing_data = np.where(cropped_img[:,:,0]==0,255,0).astype('uint8')

                contour_mask = np.zeros(missing_data.shape)

                opening = cv2.morphologyEx(missing_data.astype('uint8'), cv2.MORPH_OPEN, kernel)

                contours= cv2.findContours(opening,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

                if len(contours[0])>0:

                    largest_contour = max(contours[0], key = cv2.contourArea)

                    cv2.fillPoly(contour_mask, pts =[largest_contour], color=(255))

                    kernel = np.ones((5,5),np.uint8)

                    opening = cv2.morphologyEx(contour_mask, cv2.MORPH_OPEN, kernel)

                    fixed_mask = np.where(opening[:,:]==255,0,255)

                    rectangular_mask[roi[0]:roi[2], roi[1]:roi[3]] = fixed_mask.copy()

                    

                if mask_dict[category_list[class_id-1]]==[]:

                    mask_dict[category_list[class_id-1]] = rectangular_mask

                else:

                    previous_mask = mask_dict[category_list[class_id-1]].copy()

                    #prevents a bug where the mask is in int64

                    previous_mask = previous_mask.astype('float64')

                    boolean_mask = np.ma.mask_or(previous_mask, rectangular_mask)

                    merged_mask = np.where(boolean_mask, 255, 0)

                    mask_dict[category_list[class_id-1]] = merged_mask



            

            #Going through the masks per category and create a md mask in RLE

            for cloud_category in mask_dict.keys():

                if mask_dict[cloud_category]!=[]:

                    #resizing for submission

                    resized_mask = cv2.resize((mask_dict[cloud_category]/255).astype('uint8'), dsize=(525,350), interpolation = cv2.INTER_LINEAR)

                    rle_str = rle_encoding(resized_mask)

                    image_label = "{}_{}".format(image_id,cloud_category)

                    submission_df.loc[submission_df['Image_Label']==image_label,'EncodedPixels'] = rle_str

        else:

            masks, rois = r['masks'], r['rois']
submission_df.query("EncodedPixels!=''").head()
submission_df.to_csv("../../working/submission.csv",index=False)