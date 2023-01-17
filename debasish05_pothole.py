!pip install tensorflow==1.13.1
import tensorflow as tf
tf.__version__


import numpy as np 
from tqdm import tqdm
import pandas as pd 
import glob
from sklearn.model_selection import KFold
import sys
import random
import math
import cv2
import matplotlib.pyplot as plt
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

DATA_DIR = '/kaggle/input'
ROOT_DIR = '/kaggle/working'
!git clone https://www.github.com/matterport/Mask_RCNN.git
os.chdir('/kaggle/working/Mask_RCNN')

sys.path.append(os.path.join(ROOT_DIR, 'Mask_RCNN'))  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
from mrcnn.utils import Dataset
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log
from mrcnn.visualize import display_instances
from mrcnn.utils import extract_bboxes
!wget --quiet https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5
!ls -lh mask_rcnn_coco.h5

COCO_WEIGHTS_PATH = "mask_rcnn_coco.h5"

from xml.etree import ElementTree

# function to extract bounding boxes from an annotation file
def extract_boxes(filename):
    # load and parse the file
    tree = ElementTree.parse(filename)
    # get the root of the document
    root = tree.getroot()
    # extract each bounding box
    boxes = list()
    for box in root.findall('.//bndbox'):
        xmin = int(box.find('xmin').text)
        ymin = int(box.find('ymin').text)
        xmax = int(box.find('xmax').text)
        ymax = int(box.find('ymax').text)
        coors = [xmin, ymin, xmax, ymax]
        boxes.append(coors)
    # extract image dimensions
    width = int(root.find('.//size/width').text)
    height = int(root.find('.//size/height').text)
    return boxes, width, height

# extract details form annotation file
boxes, w, h = extract_boxes('/kaggle/input/annotated-potholes-dataset/annotated-images/img-1.xml')
# summarize extracted details
print(boxes, w, h)
class PothholeData(Dataset):
    # load the dataset definitions
    def load_dataset(self, dataset, is_train=True):
        # define one class
        self.add_class("dataset", 1, "pothole")
        # define data locations
        images_dir = '/kaggle/input/annotated-potholes-dataset/annotated-images/'
        annotations_dir = '/kaggle/input/annotated-potholes-dataset/annotated-images/'
        # find all images
        for filename in dataset :
            # extract image id
            #image_id = filename[:-4]
            if '.xml' in filename:
                image_id=filename.replace('.xml','')[4:]
            # skip bad images
            #if image_id in ['00090']:
            #    continue
            # skip all images after 150 if we are building the train set
            #if is_train and int(image_id) >= 150:
            #    continue
            # skip all images before 150 if we are building the test/val set
            #if not is_train and int(image_id) < 150:
            #    continue
            if 'xml' in filename:
                #print(i.replace('xml','jpg'))
                img_path = images_dir + filename.replace('xml','jpg')
            ann_path = annotations_dir + filename
            # add to dataset
            self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)

    # extract bounding boxes from an annotation file
    def extract_boxes(self, filename):
        # load and parse the file
        tree = ElementTree.parse(filename)
        # get the root of the document
        root = tree.getroot()
        # extract each bounding box
        boxes = list()
        for box in root.findall('.//bndbox'):
            xmin = int(box.find('xmin').text)
            ymin = int(box.find('ymin').text)
            xmax = int(box.find('xmax').text)
            ymax = int(box.find('ymax').text)
            coors = [xmin, ymin, xmax, ymax]
            boxes.append(coors)
        # extract image dimensions
        width = int(root.find('.//size/width').text)
        height = int(root.find('.//size/height').text)
        return boxes, width, height

    # load the masks for an image
    def load_mask(self, image_id):
        # get details of image
        info = self.image_info[image_id]
        # define box file location
        path = info['annotation']
        # load XML
        boxes, w, h = self.extract_boxes(path)
        # create one array for all masks, each on a different channel
        masks =np.zeros([h, w, len(boxes)], dtype='uint8')
        # create masks
        class_ids = list()
        for i in range(len(boxes)):
            box = boxes[i]
            row_s, row_e = box[1], box[3]
            col_s, col_e = box[0], box[2]
            masks[row_s:row_e, col_s:col_e, i] = 1
            class_ids.append(self.class_names.index('pothole'))
        return masks, np.asarray(class_ids, dtype='int32')

    # load an image reference
    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path']
import json

f=open('/kaggle/input/annotated-potholes-dataset/splits.json',)
traindata=json.load(f)['train']
f=open('/kaggle/input/annotated-potholes-dataset/splits.json',)
testdata=json.load(f)['test']
train_set = PothholeData()
train_set.load_dataset(traindata, is_train=True)
train_set.prepare()
print('Train: %d' % len(train_set.image_ids))
test_set = PothholeData()
test_set.load_dataset(testdata, is_train=False)
test_set.prepare()
print('Test: %d' % len(test_set.image_ids))
image_id = 200
image = train_set.load_image(image_id)
print(image.shape)
mask, class_ids = train_set.load_mask(image_id)
print(mask.shape)
plt.imshow(image)
# plot mask
plt.imshow(mask[:, :, 0], cmap='gray', alpha=0.5)
plt.show()

for image_id in train_set.image_ids:
    # load image info
    info = train_set.image_info[image_id]
    # display on the console
    print(info)
bbox = extract_bboxes(mask)
# display image with masks and bounding boxes
display_instances(image, bbox, mask, class_ids, train_set.class_names)
# define a configuration for the model
class Potholeconfig(Config):
    # Give the configuration a recognizable name
    NAME = "sonu_cnfg"
    #NAME = 'pneumonia'
    
    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 8
    
    BACKBONE = 'resnet50'
    
    NUM_CLASSES = 2  # background + 1 pneumonia classes
    
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256
    #RPN_ANCHOR_SCALES = (16, 32, 64, 128)
    TRAIN_ROIS_PER_IMAGE = 32
    MAX_GT_INSTANCES = 4
    DETECTION_MAX_INSTANCES = 3
    DETECTION_MIN_CONFIDENCE = 0.78  ## match target distribution
    DETECTION_NMS_THRESHOLD = 0.01

    STEPS_PER_EPOCH = 200


# prepare config
config = Potholeconfig()
from mrcnn.model import MaskRCNN
!pip install 'tensorflow-gpu==1.13.1'
positive_indices=[]
try:
    moderl = MaskRCNN(mode='training', model_dir='./', config=config)
except AttributeError:
    positive_indices = tf.random.shuffle(positive_indices)[:positive_count]
    # load weights (mscoco) and exclude the output layers
model.load_weights('mask_rcnn_coco.h5', by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])
# train weights (output layers or 'heads')
#model.keras_model.metrics_tensors = [] 
model.train(train_set, test_set, learning_rate=config.LEARNING_RATE,epochs=1, layers='heads')
