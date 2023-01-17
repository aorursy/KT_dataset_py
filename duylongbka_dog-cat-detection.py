# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os 

import sys

import random

import math

import numpy as np

import matplotlib.pyplot as plt

DATA_DIR = '/kaggle/input'



# Directory to save logs and trained model

ROOT_DIR = '/kaggle/working'
!git clone https://www.github.com/matterport/Mask_RCNN.git

os.chdir('Mask_RCNN')

!python setup.py -q install
sys.path.append(os.path.join(ROOT_DIR, 'Mask_RCNN'))

from xml.etree import ElementTree

from mrcnn.utils import Dataset

from numpy import zeros, asarray, expand_dims, mean

from matplotlib import pyplot

from mrcnn.utils import extract_bboxes, compute_ap

from mrcnn.visualize import display_instances

from mrcnn.config import Config

from mrcnn.model import MaskRCNN, load_image_gt, mold_image

from os import listdir
cat_classes = ['Abyssinian', 'Bombay', 'Egyptian_Mau', 'Sphynx']

dog_classes = [ 'shiba_inu', 'american_bulldog', 'boxer', 'yorkshire_terrier']

class CatDataset(Dataset):

    def load_dataset(self, dataset_dir, is_train=True):

        images_dir = dataset_dir + '/images/images/'

        annotations_dir = dataset_dir + '/annotations/annotations/xmls/'

        self.add_class("dataset", 1, 'dog')

        self.add_class("dataset", 2, 'cat')

        dataset_image_id = 1

        for filename in listdir(images_dir):

            annot_path = annotations_dir + filename.replace('.jpg', '.xml')

            animal_type = filename[0 : filename.rindex('_')]

            if ( animal_type in dog_classes or animal_type in cat_classes) and os.path.isfile(annot_path):

                image_id = filename.split('_')[-1].replace('.jpg', '')

                if is_train and int(image_id) >= 170:

                    continue

                if not is_train and int(image_id) < 170:

                    continue

                image_path = images_dir + filename

                self.add_image('dataset', image_id=dataset_image_id, path=image_path, annotation=annot_path)

                dataset_image_id += 1 

    

    def extract_boxes(self, filepath):

        tree = ElementTree.parse(filepath)

        root = tree.getroot()

        boxes = list()

        for box in root.findall('.//bndbox'):

            xmin = int(box.find('xmin').text)

            ymin = int(box.find('ymin').text)

            xmax = int(box.find('xmax').text)

            ymax = int(box.find('ymax').text)

            boxes.append([xmin, ymin, xmax, ymax])

        file_name = root.find('.//filename').text

        image_class_name = file_name[0: file_name.rindex('_')]

        width = int(root.find('.//size/width').text)

        height = int(root.find('.//size/height').text)

        return boxes, width, height, image_class_name



    def load_mask(self, image_id):

        info = self.image_info[image_id]

        path = info['annotation']

        boxes, width, height, image_class_name = self.extract_boxes(path)

        masks = zeros([height, width, len(boxes)], dtype='uint8')

        class_ids = list()

        for i in range(len(boxes)):

            box = boxes[i]

            row_s, row_e = box[1], box[3]

            col_s, col_e = box[0], box[2]

            masks[row_s:row_e, col_s: col_e, i] = 1

        if image_class_name in dog_classes:

            class_ids.append(self.class_names.index('dog'))

        elif image_class_name in cat_classes:

            class_ids.append(self.class_names.index('cat'))

        return masks, asarray(class_ids, dtype='int32')

    def image_reference(self, image_id):

        info = self.image_info(image_id)
from mrcnn.visualize import display_instances

from mrcnn.utils import extract_bboxes



train_set = CatDataset()

train_set.load_dataset('/kaggle/input/the-oxfordiiit-pet-dataset', is_train=True)

train_set.prepare()



test_set = CatDataset()

test_set.load_dataset('/kaggle/input/the-oxfordiiit-pet-dataset', is_train=False)

test_set.prepare()





class AnimalConfig(Config):

    NAME = 'cat_cfg'

    NUM_CLASSES = 3

    STEPS_PER_EPOCH = 581

    SAVE_BEST_ONLY = True



config = AnimalConfig()

config.display()



model = MaskRCNN(mode='training', model_dir='./', config = config)

model.load_weights('/kaggle/input/mrcnn-assets/mrcnntools/mask_rcnn_coco.h5', by_name=True, exclude=["mrcnn_class_logits","mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])

model.train(train_set, test_set, learning_rate=config.LEARNING_RATE, epochs=5, layers='heads')
from os import listdir

from xml.etree import ElementTree

from numpy import zeros

from numpy import asarray

from numpy import expand_dims

from matplotlib import pyplot

from matplotlib.patches import Rectangle

from mrcnn.config import Config

from mrcnn.model import MaskRCNN

from mrcnn.model import mold_image

from mrcnn.utils import Dataset



class PredictConfig(Config):

    NAME = 'cat_cfg'

    NUM_CLASSES = 3

    # simplify GPU config

    GPU_COUNT = 1

    IMAGES_PER_GPU = 1



class_labels = ['', 'dog', 'cat']

# plot a number of photos with ground truth and predictions

def plot_actual_vs_predicted(dataset, model, cfg, n_images=5):

    # load image and mask

    for i in range(n_images):

        # load the image and mask

        image = dataset.load_image(i)

        # convert pixel values (e.g. center)

        scaled_image = mold_image(image, cfg)

        # convert image into one sample

        sample = expand_dims(scaled_image, 0)

        # make prediction

        yhat = model.detect(sample, verbose=0)[0]

        # plot raw pixel data

        pyplot.imshow(image)

        if i == 0:

            pyplot.title('Predicted')

        ax = pyplot.gca()

        # draw text and score in top left corner

        class_ids = yhat['class_ids']

        scores = yhat['scores']

        # plot each box

        box_id = 0

        for box in yhat['rois']:

            # get coordinates

            y1, x1, y2, x2 = box

            # calculate width and height of the box

            width, height = x2 - x1, y2 - y1

            # create the shape

            rect = Rectangle((x1, y1), width, height, fill=False, color='red')

            # draw the box

            ax.add_patch(rect)

            label = "%s (%.3f)" % (class_labels[class_ids[box_id]], scores[box_id])

            box_id += 1

            pyplot.text(x1, y1, label, color='red')

        # show the figure

        pyplot.show()



train_set = CatDataset()

train_set.load_dataset('/kaggle/input/the-oxfordiiit-pet-dataset', is_train=True)

train_set.prepare()



test_set = CatDataset()

test_set.load_dataset('/kaggle/input/the-oxfordiiit-pet-dataset', is_train=False)

test_set.prepare() 





config = PredictConfig()

config.display()



infer_model = MaskRCNN(mode='inference', model_dir='./', config=config)



model_path = infer_model.find_last()



infer_model.load_weights(model_path, by_name=True)

# plot predictions for train dataset

plot_actual_vs_predicted(test_set, infer_model, config, 20)
