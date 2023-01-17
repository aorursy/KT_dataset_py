import os 

import sys

import random

import math

import numpy as np

import cv2

import matplotlib.pyplot as plt

import json

import pydicom

from imgaug import augmenters as iaa

from tqdm import tqdm

import pandas as pd 

import glob 



ROOT_DIR = os.path.abspath("../")



path="../Mask_RCNN"

os.mkdir(path)

#!git clone git+https://www.github.com/matterport/Mask_RCNN.git

!pip install https://www.github.com/matterport/Mask_RCNN.git

os.chdir('../Mask_RCNN')



# Import Mask RCNN

path=sys.path.append(os.path.join(ROOT_DIR, 'Mask_RCNN'))  # To find local version of the library

from mrcnn.config import Config

from mrcnn import utils

import mrcnn.model as modellib

from mrcnn import visualize

from mrcnn.model import log



train_dicom_dir = os.path.join(ROOT_DIR, 'train_images')

test_dicom_dir = os.path.join(ROOT_DIR, 'test_images')



COCO_WEIGHTS_PATH = os.path.join(os.path.abspath("../input"), "mask_rcnn_coco.h5")

DEFAULT_LOGS_DIR = os.path.join(os.path.abspath("../input"), "logs")

RESULTS_DIR = os.path.join(os.path.abspath("../input"), "results")

VAL_IMAGE_IDS = pd.read_csv(os.path.abspath("../input/train_masks.csv")).ImageId



class NucleusConfig(Config):

    NAME = "nucleus"



    IMAGES_PER_GPU = 6

    NUM_CLASSES = 1 + 1



 

    STEPS_PER_EPOCH = (657 - len(VAL_IMAGE_IDS)) // IMAGES_PER_GPU

    VALIDATION_STEPS = max(1, len(VAL_IMAGE_IDS)) // IMAGES_PER_GPU



    DETECTION_MIN_CONFIDENCE = 0

    BACKBONE = "resnet50"



    IMAGE_RESIZE_MODE = "crop"

    IMAGE_MIN_DIM = 512

    IMAGE_MAX_DIM = 512

    IMAGE_MIN_SCALE = 2.0



    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)



    POST_NMS_ROIS_TRAINING = 1000

    POST_NMS_ROIS_INFERENCE = 2000



    RPN_NMS_THRESHOLD = 0.9



    RPN_TRAIN_ANCHORS_PER_IMAGE = 64



    MEAN_PIXEL = np.array([43.53, 39.56, 48.22])



    USE_MINI_MASK = True

    MINI_MASK_SHAPE = (56, 56)  



    TRAIN_ROIS_PER_IMAGE = 128



    MAX_GT_INSTANCES = 200



    DETECTION_MAX_INSTANCES = 400





class NucleusInferenceConfig(NucleusConfig):

    GPU_COUNT = 1

    IMAGES_PER_GPU = 1

    IMAGE_RESIZE_MODE = "pad64"

    RPN_NMS_THRESHOLD = 0.7





class NucleusDataset(utils.Dataset):



    def load_nucleus(self, dataset_dir, subset):

        self.add_class("nucleus", 1, "nucleus")

        assert subset in ["train", "val", "stage1_train", "stage1_test", "stage2_test"]

        subset_dir = "stage1_train" if subset in ["train", "val"] else subset

        dataset_dir = os.path.join(dataset_dir, subset_dir)

        if subset == "val":

            image_ids = VAL_IMAGE_IDS

        else:

            image_ids = next(os.walk(dataset_dir))[1]

            if subset == "train":

                image_ids = list(set(image_ids) - set(VAL_IMAGE_IDS))

        for image_id in image_ids:

            self.add_image(

                "nucleus",

                image_id=image_id,

                path=os.path.join(dataset_dir, image_id, "images/{}.png".format(image_id)))



    def load_mask(self, image_id):

        info = self.image_info[image_id]

        mask_dir = os.path.join(os.path.dirname(os.path.dirname(info['path'])), "masks")

        mask = []

        for f in next(os.walk(mask_dir))[2]:

            if f.endswith(".png"):

                m = skimage.io.imread(os.path.join(mask_dir, f)).astype(np.bool)

                mask.append(m)

        mask = np.stack(mask, axis=-1)

        return mask, np.ones([mask.shape[-1]], dtype=np.int32)



    def image_reference(self, image_id):

        info = self.image_info[image_id]

        if info["source"] == "nucleus":

            return info["id"]

        else:

            super(self.__class__, self).image_reference(image_id)





def train(model, dataset_dir, subset):

    dataset_train = NucleusDataset()

    dataset_train.load_nucleus(dataset_dir, subset)

    dataset_train.prepare()



    dataset_val = NucleusDataset()

    dataset_val.load_nucleus(dataset_dir, "val")

    dataset_val.prepare()



    augmentation = iaa.SomeOf((0, 2), [

        iaa.Fliplr(0.5),

        iaa.Flipud(0.5),

        iaa.OneOf([iaa.Affine(rotate=90),

                   iaa.Affine(rotate=180),

                   iaa.Affine(rotate=270)]),

        iaa.Multiply((0.8, 1.5)),

        iaa.GaussianBlur(sigma=(0.0, 5.0))

    ])



    model.train(dataset_train, dataset_val,

                learning_rate=config.LEARNING_RATE,

                epochs=20,

                augmentation=augmentation,

                layers='heads')



    model.train(dataset_train, dataset_val,

                learning_rate=config.LEARNING_RATE,

                epochs=40,

                augmentation=augmentation,

                layers='all')



def rle_encode(mask):

    mask.ndim == 2 

    m = mask.T.flatten()

    g = np.diff(np.concatenate([[0], m, [0]]), n=1)

    rle = np.where(g != 0)[0].reshape([-1, 2]) + 1

    rle[:, 1] = rle[:, 1] - rle[:, 0]

    return " ".join(map(str, rle.flatten()))





def rle_decode(rle, shape):

    rle = list(map(int, rle.split()))

    rle = np.array(rle, dtype=np.int32).reshape([-1, 2])

    rle[:, 1] += rle[:, 0]

    rle -= 1

    mask = np.zeros([shape[0] * shape[1]], np.bool)

    for s, e in rle:

        assert 0 <= s < mask.shape[0]

        assert 1 <= e <= mask.shape[0], "shape: {}  s {}  e {}".format(shape, s, e)

        mask[s:e] = 1

    mask = mask.reshape([shape[1], shape[0]]).T

    return mask





def mask_to_rle(image_id, mask, scores):

    mask.ndim == 3

    if mask.shape[-1] == 0:

        return "{},".format(image_id)

    order = np.argsort(scores)[::-1] + 1 

    mask = np.max(mask * np.reshape(order, [1, 1, -1]), -1)

    lines = []

    for o in order:

        m = np.where(mask == o, 1, 0)

        if m.sum() == 0.0:

            continue

        rle = rle_encode(m)

        lines.append("{}, {}".format(image_id, rle))

    return "\n".join(lines)







def detect(model, dataset_dir, subset):

    print("Running on {}".format(dataset_dir))



    if not os.path.exists(RESULTS_DIR):

        os.makedirs(RESULTS_DIR)

    submit_dir = "submit_{:%Y%m%dT%H%M%S}".format(datetime.datetime.now())

    submit_dir = os.path.join(RESULTS_DIR, submit_dir)

    os.makedirs(submit_dir)



    dataset = NucleusDataset()

    dataset.load_nucleus(dataset_dir, subset)

    dataset.prepare()

    submission = []

    for image_id in dataset.image_ids:

        image = dataset.load_image(image_id)

        r = model.detect([image], verbose=0)[0]

        source_id = dataset.image_info[image_id]["id"]

        rle = mask_to_rle(source_id, r["masks"], r["scores"])

        submission.append(rle)

        visualize.display_instances(

            image, r['rois'], r['masks'], r['class_ids'],

            dataset.class_names, r['scores'],

            show_bbox=False, show_mask=False,

            title="Predictions")

        plt.savefig("{}/{}.png".format(submit_dir, dataset.image_info[image_id]["id"]))



    

    submission = "ImageId,EncodedPixels\n" + "\n".join(submission)

    file_path = os.path.join(submit_dir, "submission.csv")

    with open(file_path, "w") as f:

        f.write(submission)
#!pip install git+https://www.github.com/matterport/Mask_RCNN.git


