import os

import gc

import sys

import json

import glob

import random

from pathlib import Path



import cv2

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



import itertools

from tqdm import tqdm



from imgaug import augmenters as iaa

from sklearn.model_selection import train_test_split



import warnings 

warnings.filterwarnings("ignore")
DATA_DIR = Path('/kaggle/input')

ROOT_DIR = Path('/kaggle/working')



NUM_CATS = 46 # ในที่นี้จะทำ classification เฉพาะส่วนที่เป็น category ซึ่งมีจำนวนทั้งสิ้น 46

IMAGE_SIZE = 512 # รูปภาพที่เป็น input เข้า Mask R-CNN จะถูกทำให้เป็นขนาด 512x512 ซึ่งเท่ากับขนาดของ mask ที่ส่งเป็นคำตอบ
!git clone https://www.github.com/matterport/Mask_RCNN.git

os.chdir('Mask_RCNN')



!rm -rf .git

!rm -rf images assets
sys.path.append(ROOT_DIR/'Mask_RCNN')

from mrcnn.config import Config

from mrcnn import utils

import mrcnn.model as modellib

from mrcnn import visualize

from mrcnn.model import log
!wget --quiet https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5

!ls -lh mask_rcnn_coco.h5



COCO_WEIGHTS_PATH = 'mask_rcnn_coco.h5'
class FashionConfig(Config):

    NAME = "fashion"

    NUM_CLASSES = NUM_CATS + 1 # +1 สำหรับคลาส background

    

    GPU_COUNT = 1

    IMAGES_PER_GPU = 4

    

    BACKBONE = 'resnet50'

    

    IMAGE_MIN_DIM = IMAGE_SIZE

    IMAGE_MAX_DIM = IMAGE_SIZE    

    IMAGE_RESIZE_MODE = 'none'

    

    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)

    TRAIN_ROIS_PER_IMAGE = 100

    

    # เนื่องจาก Kaggle จำกัดเวลาในการรัน kernel ไว้ 9 ชั่วโมง 

    # เราจึงกำหนดค่า STEPS_PER_EPOCH และ VALIDATION_STEPS ให้รันทันในเวลานี้ครับ

    STEPS_PER_EPOCH = 5500

    VALIDATION_STEPS = 100

    

config = FashionConfig()

config.display()
with open(DATA_DIR/"label_descriptions.json") as f:

    label_descriptions = json.load(f)



label_names = [x['name'] for x in label_descriptions['categories']]
segment_df = pd.read_csv(DATA_DIR/"train.csv")

segment_df['CategoryId'] = segment_df['ClassId'].str.split('_').str[0]



print("Total segments: ", len(segment_df))

segment_df.head()
image_df = segment_df.groupby('ImageId')['EncodedPixels', 'CategoryId'].agg(lambda x: list(x))

size_df = segment_df.groupby('ImageId')['Height', 'Width'].mean()

image_df = image_df.join(size_df, on='ImageId')



print("Total images: ", len(image_df))

image_df.head()
def resize_image(image_path):

    img = cv2.imread(image_path)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)  

    return img
class FashionDataset(utils.Dataset):



    def __init__(self, df):

        super().__init__(self)

        

        # Add classes

        for i, name in enumerate(label_names):

            self.add_class("fashion", i+1, name)

        

        # Add images 

        for i, row in df.iterrows():

            self.add_image("fashion", 

                           image_id=row.name, 

                           path=str(DATA_DIR/'train'/row.name), 

                           labels=row['CategoryId'],

                           annotations=row['EncodedPixels'], 

                           height=row['Height'], width=row['Width'])



    def image_reference(self, image_id):

        info = self.image_info[image_id]

        return info['path'], [label_names[int(x)] for x in info['labels']]

    

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
dataset = FashionDataset(image_df)

dataset.prepare()



for i in range(6):

    image_id = random.choice(dataset.image_ids)

    print(dataset.image_reference(image_id))

    

    image = dataset.load_image(image_id)

    mask, class_ids = dataset.load_mask(image_id)

    visualize.display_top_masks(image, mask, class_ids, dataset.class_names, limit=4)
train_df, valid_df = train_test_split(image_df, test_size=0.2, random_state=42)



train_dataset = FashionDataset(train_df)

train_dataset.prepare()



valid_dataset = FashionDataset(valid_df)

valid_dataset.prepare()
train_segments = np.concatenate(train_df['CategoryId'].values).astype(int)

print("Total train images: ", len(train_df))

print("Total train segments: ", len(train_segments))



plt.figure(figsize=(12, 3))

values, counts = np.unique(train_segments, return_counts=True)

plt.bar(values, counts)

plt.xticks(values, label_names, rotation='vertical')

plt.show()



valid_segments = np.concatenate(valid_df['CategoryId'].values).astype(int)

print("Total validation images: ", len(valid_df))

print("Total validation segments: ", len(valid_segments))



plt.figure(figsize=(12, 3))

values, counts = np.unique(valid_segments, return_counts=True)

plt.bar(values, counts)

plt.xticks(values, label_names, rotation='vertical')

plt.show()
model = modellib.MaskRCNN(mode='training', config=config, model_dir=ROOT_DIR)



model.load_weights(COCO_WEIGHTS_PATH, by_name=True, exclude=[

    'mrcnn_class_logits', 'mrcnn_bbox_fc', 'mrcnn_bbox', 'mrcnn_mask'])
augmentation = iaa.Sequential([

    iaa.Fliplr(0.5)

])
%%time

model.train(train_dataset, valid_dataset,

            learning_rate=2e-3,

            epochs=1,

            layers='heads',

            augmentation=None)



history = model.keras_model.history.history
%%time

model.train(train_dataset, valid_dataset,

            learning_rate=1e-3,

            epochs=3,

            layers='all',

            augmentation=augmentation)



new_history = model.keras_model.history.history

for k in new_history: history[k] = history[k] + new_history[k]
epochs = range(3)



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
class InferenceConfig(FashionConfig):

    GPU_COUNT = 1

    IMAGES_PER_GPU = 1



inference_config = InferenceConfig()



glob_list = glob.glob(f'/kaggle/working/fashion*/mask_rcnn_fashion_{best_epoch:04d}.h5')

model_path = glob_list[0] if glob_list else ''



model = modellib.MaskRCNN(mode='inference', 

                          config=inference_config,

                          model_dir=ROOT_DIR)



assert model_path != '', "Provide path to trained weights"

print("Loading weights from ", model_path)

model.load_weights(model_path, by_name=True)
sample_df = pd.read_csv(DATA_DIR/"sample_submission.csv")

sample_df.head()
def to_rle(bits):

    rle = []

    pos = 1

    for bit, group in itertools.groupby(bits):

        group_list = list(group)

        if bit:

            rle.extend([pos, len(group_list)])

        pos += len(group_list)

    return rle
def trim_masks(masks, rois, class_ids):

    class_pos = np.argsort(class_ids)

    class_rle = to_rle(np.sort(class_ids))

    

    pos = 0

    for i, _ in enumerate(class_rle[::2]):

        previous_pos = pos

        pos += class_rle[2*i+1]

        if pos-previous_pos == 1:

            continue 

        mask_indices = class_pos[previous_pos:pos]

        

        union_mask = np.zeros(masks.shape[:-1], dtype=bool)

        for m in mask_indices:

            masks[:, :, m] = np.logical_and(masks[:, :, m], np.logical_not(union_mask))

            union_mask = np.logical_or(masks[:, :, m], union_mask)

        for m in mask_indices:

            mask_pos = np.where(masks[:, :, m]==True)

            if np.any(mask_pos):

                y1, x1 = np.min(mask_pos, axis=1)

                y2, x2 = np.max(mask_pos, axis=1)

                rois[m, :] = [y1, x1, y2, x2]

            

    return masks, rois
%%time

sub_list = []

missing_count = 0

for i, row in tqdm(sample_df.iterrows(), total=len(sample_df)):

    image = resize_image(str(DATA_DIR/'test'/row['ImageId']))

    result = model.detect([image])[0]

    if result['masks'].size > 0:

        masks, _ = trim_masks(result['masks'], result['rois'], result['class_ids'])

        for m in range(masks.shape[-1]):

            mask = masks[:, :, m].ravel(order='F')

            rle = to_rle(mask)

            label = result['class_ids'][m] - 1

            sub_list.append([row['ImageId'], ' '.join(list(map(str, rle))), label])

    else:

        # The system does not allow missing ids, this is an easy way to fill them

        sub_list.append([row['ImageId'], '1 1', 23])

        missing_count += 1
submission_df = pd.DataFrame(sub_list, columns=sample_df.columns.values)

print("Total image results: ", submission_df['ImageId'].nunique())

print("Missing images: ", missing_count)

submission_df.head()
submission_df.to_csv("submission.csv", index=False)
for i in range(9):

    image_id = sample_df.sample()['ImageId'].values[0]

    image_path = str(DATA_DIR/'test'/image_id)

    

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

        

        masks, rois = trim_masks(masks, rois, r['class_ids'])

    else:

        masks, rois = r['masks'], r['rois']

        

    visualize.display_instances(img, rois, masks, r['class_ids'], 

                                ['bg']+label_names, r['scores'],

                                title=image_id, figsize=(12, 12))