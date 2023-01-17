import os



import numpy as np

import pandas as pd



import pydicom

import cv2

import matplotlib.pyplot as plt



from keras.models import *

from keras.layers import *

from keras.optimizers import *

from keras.losses import binary_crossentropy

from keras.utils import Sequence

from keras import backend as keras

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ModelCheckpoint, LearningRateScheduler



from glob import glob

from tqdm import tqdm
INPUT_DIR = os.path.join("..", "input")



SEGMENTATION_DIR = os.path.join(INPUT_DIR, "u-net-lung-segmentation-montgomery-shenzhen")

SEGMENTATION_MODEL = os.path.join(SEGMENTATION_DIR, "unet_lung_seg.hdf5")

SEGMENTATION_RESULT = "segmentation"

SEGMENTATION_RESULT_TRAIN = os.path.join(SEGMENTATION_RESULT, "train")

SEGMENTATION_RESULT_TEST = os.path.join(SEGMENTATION_RESULT, "test")



RSNA_DIR = os.path.join(INPUT_DIR, "rsna-pneumonia-detection-challenge")

RSNA_TRAIN_DIR = os.path.join(RSNA_DIR, "stage_1_train_images")

RSNA_TEST_DIR = os.path.join(RSNA_DIR, "stage_1_test_images")

RSNA_LABELS_FILE = os.path.join(RSNA_DIR, "stage_1_train_labels.csv")

RSNA_CLASS_INFO_FILE = os.path.join(RSNA_DIR, "stage_1_detailed_class_info.csv")
!mkdir segmentation

!mkdir segmentation/train

!mkdir segmentation/test
def dice_coef(y_true, y_pred):

    y_true_f = keras.flatten(y_true)

    y_pred_f = keras.flatten(y_pred)

    intersection = keras.sum(y_true_f * y_pred_f)

    return (2. * intersection + 1) / (keras.sum(y_true_f) + keras.sum(y_pred_f) + 1)



def dice_coef_loss(y_true, y_pred):

    return -dice_coef(y_true, y_pred)



segmentation_model = load_model(SEGMENTATION_MODEL, \

                                custom_objects={'dice_coef_loss': dice_coef_loss, \

                                                'dice_coef': dice_coef})



segmentation_model.summary()
def image_to_train(img):

    npy = img / 255

    npy = np.reshape(npy, npy.shape + (1,))

    npy = np.reshape(npy,(1,) + npy.shape)

    return npy



def train_to_image(npy):

    img = (npy[0,:, :, 0] * 255.).astype(np.uint8)

    return img
def segment_image(pid, img, save_to):

    img = cv2.resize(img, (512, 512))

    segm_ret = segmentation_model.predict(image_to_train(img), \

                                          verbose=0)



    img = cv2.bitwise_and(img, img, mask=train_to_image(segm_ret))

    

    cv2.imwrite(os.path.join(save_to, "%s.png" % pid), img)



for filename in tqdm(glob(os.path.join(RSNA_TRAIN_DIR, "*.dcm"))):

    pid, fileext = os.path.splitext(os.path.basename(filename))

    img = pydicom.dcmread(filename).pixel_array

    segment_image(pid, img, SEGMENTATION_RESULT_TRAIN)



for filename in tqdm(glob(os.path.join(RSNA_TEST_DIR, "*.dcm"))):

    pid, fileext = os.path.splitext(os.path.basename(filename))

    img = pydicom.dcmread(filename).pixel_array

    segment_image(pid, img, SEGMENTATION_RESULT_TEST)
!tar zcf segmentation.tgz --directory=segmentation .

!rm -rf segmentation