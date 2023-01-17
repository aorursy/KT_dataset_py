import numpy as np

import pandas as pd



import os

import sys

from tqdm import tqdm

from pathlib import Path

import tensorflow as tf

import skimage.io

import matplotlib.pyplot as plt
# https://www.kaggle.com/pednoi/training-mask-r-cnn-to-be-a-fashionista-lb-0-07



!git clone https://www.github.com/matterport/Mask_RCNN.git

os.chdir('Mask_RCNN')



!rm -rf .git # to prevent an error when the kernel is committed

!rm -rf images assets # to prevent displaying images at the bottom of a kernel
DATA_DIR = Path('/kaggle/input')

ROOT_DIR = Path('/kaggle/working')
sys.path.append(ROOT_DIR/'Mask_RCNN')
!pip install pycocotools
from mrcnn.config import Config

from mrcnn import utils

import mrcnn.model as modellib

from mrcnn import visualize

from mrcnn.model import log
!wget https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5
COCO_MODEL_PATH = 'mask_rcnn_coco.h5'
# Import COCO config

sys.path.append(os.path.join(ROOT_DIR, "Mask_RCNN/samples/coco/"))  # To find local version

import coco
class InferenceConfig(coco.CocoConfig):

    # Set batch size to 1 since we'll be running inference on

    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU

    GPU_COUNT = 1

    IMAGES_PER_GPU = 1

    IMAGE_MIN_DIM = 256

    IMAGE_MAX_DIM = 256

    

config = InferenceConfig()

config.display()
# Create model object in inference mode.

model = modellib.MaskRCNN(mode="inference", config=config, model_dir=ROOT_DIR)



# Load weights trained on MS-COCO

model.load_weights(COCO_MODEL_PATH, by_name=True)
# COCO Class names

# Index of the class in the list is its ID. For example, to get ID of

# the teddy bear class, use: class_names.index('teddy bear')

class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',

               'bus', 'train', 'truck', 'boat', 'traffic light',

               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',

               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',

               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',

               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',

               'kite', 'baseball bat', 'baseball glove', 'skateboard',

               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',

               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',

               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',

               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',

               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',

               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',

               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',

               'teddy bear', 'hair drier', 'toothbrush']
IMAGE_DIR = "/kaggle/input/test/"
os.chdir('/kaggle/')
os.listdir("./input")
!wget https://storage.googleapis.com/openimages/challenge_2019/challenge-2019-classes-description-segmentable.csv
class_lookup_df = pd.read_csv("./challenge-2019-classes-description-segmentable.csv", header=None)

empty_submission_df = pd.read_csv("input/sample_empty_submission.csv")
# we have to convert coco classes to this competition's one.



class_lookup_df.columns = ["encoded_label","label"]

class_lookup_df['label'] = class_lookup_df['label'].str.lower()

class_lookup_df.head()
empty_submission_df.head()
sample_image = "80155d58d0ee19bd.jpg"

image = skimage.io.imread(os.path.join(IMAGE_DIR, sample_image))

results = model.detect([image], verbose=1)



# Visualize results

r = results[0]

print( class_names[r['class_ids'][0]])



visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
r['masks'].shape
plt.imshow(r['masks'][:,:,0])
import base64

import numpy as np

from pycocotools import _mask as coco_mask

import typing as t

import zlib



def encode_binary_mask(mask: np.ndarray) -> t.Text:

    """Converts a binary mask into OID challenge encoding ascii text."""



    # check input mask --

    if mask.dtype != np.bool:

        raise ValueError("encode_binary_mask expects a binary mask, received dtype == %s" % mask.dtype)



    mask = np.squeeze(mask)

    if len(mask.shape) != 2:

        raise ValueError("encode_binary_mask expects a 2d mask, received shape == %s" % mask.shape)



    # convert input mask to expected COCO API input --

    mask_to_encode = mask.reshape(mask.shape[0], mask.shape[1], 1)

    mask_to_encode = mask_to_encode.astype(np.uint8)

    mask_to_encode = np.asfortranarray(mask_to_encode)



    # RLE encode mask --

    encoded_mask = coco_mask.encode(mask_to_encode)[0]["counts"]



    # compress and base64 encoding --

    binary_str = zlib.compress(encoded_mask, zlib.Z_BEST_COMPRESSION)

    base64_str = base64.b64encode(binary_str)

    return base64_str
ImageID_list = []

ImageWidth_list = []

ImageHeight_list = []

PredictionString_list = []



for num, row in tqdm(empty_submission_df.iterrows(), total=len(empty_submission_df)):

    filename = row["ImageID"] + ".jpg"

   

    image = skimage.io.imread(os.path.join(IMAGE_DIR, filename))

    results = model.detect([image])

    r = results[0]

    

    height = image.shape[0]

    width  = image.shape[1]

        

    PredictionString = ""

    

    for i in range(len(r["class_ids"])):        

        class_id = r["class_ids"][i]

        roi = r["rois"][i]

        mask = r["masks"][:,:,i]

        confidence = r["scores"][i]

        

        encoded_mask = encode_binary_mask(mask)

        

        labelname = class_names[r['class_ids'][0]]

        if class_lookup_df[class_lookup_df["label"] == labelname].shape[0] == 0:

            # no match label

            continue

        

        encoded_label = class_lookup_df[class_lookup_df["label"] == labelname]["encoded_label"].item()



        PredictionString += encoded_label 

        PredictionString += " "

        PredictionString += str(confidence)

        PredictionString += " "

        PredictionString += encoded_mask.decode()

        PredictionString += " "

        

    ImageID_list.append(row["ImageID"])

    ImageWidth_list.append(width)

    ImageHeight_list.append(height)

    PredictionString_list.append(PredictionString)
results=pd.DataFrame({"ImageID":ImageID_list,

                      "ImageWidth":ImageWidth_list,

                      "ImageHeight":ImageHeight_list,

                      "PredictionString":PredictionString_list

                     })
results.head()
results.shape
os.chdir('/kaggle/working')
results.to_csv("submission.csv", index=False)