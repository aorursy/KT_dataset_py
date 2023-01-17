# importing library to handle files

import os

from os import listdir



# importing library to handle runtime manipulation

import sys



# importing library to parse annotations

from xml.etree import ElementTree



# importing library to deal with numeric arrays

import numpy

from numpy import zeros

from numpy import asarray

from numpy import expand_dims
# installing tensorflow 1.15 to ensure compatibility with Mask R-CNN

!pip install tensorflow==1.15



import tensorflow as tf



print(tf.__version__)
# installing keras 2.1.0 to ensure compatibility with Mask R-CNN

!pip install keras==2.1.0



import keras



print(keras.__version__)
# cloning Mask R-CNN from Github

!git clone https://www.github.com/matterport/Mask_RCNN.git

os.chdir('Mask_RCNN')



# directory to save logs and trained model

ROOT_DIR = '/kaggle/working'

sys.path.append(os.path.join(ROOT_DIR, 'Mask_RCNN'))



# importing Mask R-CNN

from mrcnn.config import Config

from mrcnn import utils

from mrcnn.model import mold_image

from mrcnn.utils import Dataset

import mrcnn.model as modellib

from mrcnn import visualize

from mrcnn.model import log
# class that defines and loads the kangaroo dataset

class KangarooDataset(Dataset):

    

    # loading the dataset definitions

    def load_dataset(self, dataset_dir, is_train=True):

        

        # defining one class

        self.add_class("dataset", 1, "kangaroo")

        

        # defining data locations

        images_dir = dataset_dir + '/images/'

        annotations_dir = dataset_dir + '/annots/'

        

        # finding all images

        for filename in listdir(images_dir):

            

            # extracting image id

            image_id = filename[:-4]

            

            # limiting to 4 images for starter code

            if int(image_id) <=4:

            

                # skipping bad images

                if image_id in ['00090']:

                    continue

                    

                # skipping all images after 2 if we are building the train set

                if is_train and int(image_id) <= 2:

                    continue

                    

                # skipping all images before 2 if we are building the test/val set

                if not is_train and int(image_id) > 2:

                    continue

                    

                img_path = images_dir + filename

                ann_path = annotations_dir + image_id + '.xml'

                

                # adding to dataset

                self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)

            

    # loading all bounding boxes for an image

    def extract_boxes(self, filename):

        

        # loading and parsing the file

        root = ElementTree.parse(filename)

        boxes = list()

        

        # extracting each bounding box

        for box in root.findall('.//bndbox'):

            xmin = int(box.find('xmin').text)

            ymin = int(box.find('ymin').text)

            xmax = int(box.find('xmax').text)

            ymax = int(box.find('ymax').text)

            coors = [xmin, ymin, xmax, ymax]

            boxes.append(coors)

            

        # extracting image dimensions

        width = int(root.find('.//size/width').text)

        height = int(root.find('.//size/height').text)

        

        return boxes, width, height

    

    # loading the masks for an image

    def load_mask(self, image_id):

        

        # getting details of image

        info = self.image_info[image_id]

        

        # defining box file location

        path = info['annotation']

        

        # loading XML

        boxes, w, h = self.extract_boxes(path)

        

        # creating one array for all masks, each on a different channel

        masks = zeros([h, w, len(boxes)], dtype='uint8')

        

        # creating masks

        class_ids = list()

        

        for i in range(len(boxes)):

            box = boxes[i]

            row_s, row_e = box[1], box[3]

            col_s, col_e = box[0], box[2]

            masks[row_s:row_e, col_s:col_e, i] = 1

            class_ids.append(self.class_names.index('kangaroo'))

            

        return masks, asarray(class_ids, dtype='int32')

 

    # loading an image reference

    def image_reference(self, image_id):

        

        info = self.image_info[image_id]

        

        return info['path']
# class that defines the configuration

class KangarooConfig(Config):

    

    # defining the name of the configuration

    NAME = "kangaroo_cfg"

    

    # number of classes (background + kangaroo)

    NUM_CLASSES = 1 + 1

    

    # number of training steps per epoch, images per GPU, bacth size and validation steps

    STEPS_PER_EPOCH = 2

    IMAGES_PER_GPU = 1

    BATCH_SIZE = 1

    VALIDATION_STEPS = 2
# downloading pre-trained weights for Mask R-CNN

!wget --quiet https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5

!ls -lh mask_rcnn_coco.h5



COCO_WEIGHTS_PATH = "mask_rcnn_coco.h5"
# preparing train set

train_set = KangarooDataset()

train_set.load_dataset('/kaggle/input/wildlife-images-kangaroo/Wildlife_Kangaroo', is_train=True)

train_set.prepare()

print('Train: %d' % len(train_set.image_ids))



# preparing test/val set

test_set = KangarooDataset()

test_set.load_dataset('/kaggle/input/wildlife-images-kangaroo/Wildlife_Kangaroo', is_train=False)

test_set.prepare()

print('Test: %d' % len(test_set.image_ids))



# preparing config

config = KangarooConfig()

config.display()



# defining the model

model = modellib.MaskRCNN(mode='training', model_dir=ROOT_DIR, config=config)



# loading weights (mscoco) and exclude the output layers

model.load_weights('mask_rcnn_coco.h5', by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])



# training weights (output layers or 'heads')

model.train(train_set, test_set, learning_rate=config.LEARNING_RATE, epochs=1, layers='heads')