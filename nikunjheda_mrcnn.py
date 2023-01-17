# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!pip uninstall -y tensorflow
!pip install tensorflow-gpu==1.14
import tensorflow as tf
tf.test.is_gpu_available()
print(tf.__version__)


import os, cv2, re, random
import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array, load_img
from keras import layers, models, optimizers
from keras import backend as K
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, AvgPool2D, BatchNormalization, Reshape
from keras.callbacks import LearningRateScheduler
import matplotlib.pyplot as plt
tf.test.is_gpu_available()
train=pd.read_csv('/kaggle/input/face-mask-detection-dataset/train.csv')
print(train.head)
train.sort_values(by='name', ascending=True)
train.isnull().any().value_counts()
DIR='/kaggle/input/face-mask-detection-dataset/Medical mask/Medical mask/Medical Mask/images/'
images = [DIR+i for i in os.listdir(DIR)]
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)', text) ]
images.sort(key=natural_keys)
train_images = images[1698:] 
test_images= images[0:1698]
print(train_images)
from matplotlib.image import imread

folder = '/kaggle/input/face-mask-detection-dataset/Medical mask/Medical mask/Medical Mask/images/'
# plot first few images
for i in range(2):
	# define subplot
	plt.subplot(330 + 1 + i)
	# define filename
	filename = folder +'200' + str(i) + '.jpg'
	#print(filename)
	image = imread(filename)
	print(image.shape)
	plt.imshow(image)
# show the figure
plt.show()
!git clone https://github.com/matterport/Mask_RCNN.git
os.chdir('Mask_RCNN')
import mrcnn
from mrcnn.utils import Dataset
from mrcnn.model import MaskRCNN
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log
from mrcnn.visualize import display_instances
from mrcnn.utils import extract_bboxes

import numpy as np
from numpy import zeros
from numpy import asarray
import random
import cv2
import os
import time
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from keras.models import load_model
%matplotlib inline
from os import listdir
import json

with open('/kaggle/input/face-mask-detection-dataset/Medical mask/Medical mask/meta.json') as f:
    data_classes = json.load(f)
class DetectorConfig(Config):
    """Configuration for training pneumonia detection on the RSNA pneumonia dataset.
    Overrides values in the base Config class.
    """
    
    # Give the configuration a recognizable name  
    NAME = 'face_amsk'
    
    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    
    BACKBONE = 'resnet50'
    LEARNING_RATE=0.01
    NUM_CLASSES = 21
    DETECTION_MIN_CONFIDENCE = 0.9

    STEPS_PER_EPOCH = 100
    
config = DetectorConfig()
config.display()
class Face_mask_Dataset(Dataset):
    # load the dataset definitions
    def load_dataset(self, dataset_dir, is_train=True):
        
        # Add classes. We have only one class to add.
        for i in range(20):
            self.add_class('dataset', i+1, data_classes["classes"][i]["title"])
        
        # define data locations for images and annotations
        images_dir = dataset_dir + '/images/'
        annotations_dir = dataset_dir + '/annotations/'
        
        # Iterate through all files in the folder to 
        #add class, images and annotaions
        for filename in listdir(images_dir):
            
            # extract image id
            image_id = filename[:-4]
            if(image_id[-1]=='.'):
                image_id = image_id[:-1]
            # skip all images before 1801 if we are building the train set
            if is_train and int(image_id) <= 2800:
                continue
            # skip all images after 1800 if we are building the test set
            if  int(image_id) < 1800 :
                continue
            if  not is_train and int(image_id) > 2800:
                continue
            # setting image file
            img_path = images_dir + filename
            
            # setting annotations file
            ann_path = annotations_dir + filename + '.json'
            
            # adding images and annotations to dataset
            self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)
# extract bounding boxes from an annotation file
    def extract_boxes(self, filename):
        
        with open(filename) as f:
            x = json.load(f)
        boxes = list()
        box_class = list()
        for i in range(x["NumOfAnno"]):
            boxes.append(x["Annotations"][i]["BoundingBox"])
            box_class.append(x["Annotations"][i]["classname"])
        return boxes,box_class
    """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
     """
    def load_mask(self, image_id):
        # get details of image
        info = self.image_info[image_id]
        
        # define anntation  file location
        path = info['annotation']
        filename= info['path']
        
        boxes,boxes_class= self.extract_boxes(path)
        image = imread(filename)
        w=image.shape[1]
        h=image.shape[0]
        # create one array for all masks, each on a different channel
        masks = zeros([h, w, len(boxes)], dtype='uint8')
        
        # create masks
        class_ids = list()
        for i in range(len(boxes)):
            box = boxes[i]
            row_s, row_e = box[1], box[3]
            col_s, col_e = box[0], box[2]
            masks[row_s:row_e, col_s:col_e, i] = 1
            class_ids.append(self.class_names.index(boxes_class[i]))
        return masks, asarray(class_ids, dtype='int32')
    def image_reference(self, image_id):
        info = self.image_info[image_id]
        print(info)
        return info['path']
# prepare train set
train_set = Face_mask_Dataset()
train_set.load_dataset('/kaggle/input/face-mask-detection-dataset/Medical mask/Medical mask/Medical Mask', is_train=True)
train_set.prepare()
print('Train: %d' % len(train_set.image_ids))
# prepare test/val set
val_set = Face_mask_Dataset()
val_set.load_dataset('/kaggle/input/face-mask-detection-dataset/Medical mask/Medical mask/Medical Mask', is_train=False)
val_set.prepare()
print('Test: %d' % len(val_set.image_ids))
model = modellib.MaskRCNN(mode="training", config=config, model_dir='./')
from imgaug import augmenters as iaa
augmentation = iaa.SomeOf((0, 1), [
    iaa.Fliplr(0.5),
    iaa.Affine(
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        rotate=(-25, 25),
        shear=(-8, 8)
    ),
    iaa.Multiply((0.9, 1.1))
])
import warnings 
warnings.filterwarnings("ignore")
model.keras_model.metrics_tensors = []
model.train(train_set, val_set, 
            learning_rate=config.LEARNING_RATE, 
            epochs=5, 
            layers='all',
            augmentation=augmentation)
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
