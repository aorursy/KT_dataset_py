opts = {}

#opts['tf_version'] = 1.14                      # current version also works with tf 2.2

opts['imageType_train'] = '.tif'

opts['imageType_test'] = '.tif'

opts['number_of_channel'] = 3                   # Set if to '3' for RGB images and set it to '1' for grayscale images

opts['treshold'] = 0.5                          # treshold to convert the network output (stage 1) to binary masks

## input & output directories

opts['train_dir'] = '../input/segmentation-of-nuclei-in-cryosectioned-he-images/tissue images/'

opts['train_label_dir'] = '../input/segmentation-of-nuclei-in-cryosectioned-he-images/Annotator 1 (biologist)/Annotator 1 (biologist)/mask binary/'

opts['train_label_masks'] = '../input/segmentation-of-nuclei-in-cryosectioned-he-images/Annotator 1 (biologist)/Annotator 1 (biologist)/label masks modify/'

opts['train_dis_dir'] = '../input/segmentation-of-nuclei-in-cryosectioned-he-images/Annotator 1 (biologist)/Annotator 1 (biologist)/distance maps/'

opts['results_save_path'] ='/kaggle/working/images/'

opts['models_save_path'] ='/kaggle/working/models/'



opts['epoch_num_stage1'] = 30                   # number of epochs for stage 1

opts['quick_run'] = 0.01                         # step = (len(train)/batch_size) / quick_run (set it to large numbers just debugging the code)

opts['batch_size'] = 16                          # batch size

opts['random_seed_num'] = 19                    # keep it constant to be able to reproduce the results

opts['k_fold'] = 10                             # set to '1' to have no cross validation (much faster training but 2-3% degradation in performance)

opts['save_val_results'] = 1                    # set to '0' to skip saving the validation results in training

opts['init_LR'] = 0.001                         # initial learning rate for stage 1 and stage 2

opts['LR_decay_factor'] = 0.5                   # learning rate scheduler

opts['LR_drop_after_nth_epoch'] = 8            # learning rate scheduler

opts['crop_size'] = 512                         # crop size for training

opts['pretrained_model'] = 'efficientnetb0'     # future development 

opts['use_pretrained_flag'] = 0                 # if you want to use a pretrained model in the encoder set it to one



## disabeling warning msg

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'



# 0 = all messages are logged (default behavior)

# 1 = INFO messages are not printed

# 2 = INFO and WARNING messages are not printed

# 3 = INFO, WARNING, and ERROR messages are not printed

import warnings

warnings.simplefilter('ignore')

import sys

sys.stdout.flush() # resolving tqdm problem
import numpy as np

import tensorflow as tf

import math

from matplotlib.colors import rgb_to_hsv, hsv_to_rgb

import random



import keras

from keras.models import Model, load_model

from keras.layers import Input, BatchNormalization, Activation, add

from keras.layers.core import Dropout, Lambda

from keras.layers.convolutional import Conv2D, Conv2DTranspose, UpSampling2D

from keras.layers.pooling import MaxPooling2D

from keras.layers.merge import concatenate

from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras import backend as K

from keras.models import Model, load_model

from keras.callbacks import EarlyStopping, ModelCheckpoint,LearningRateScheduler,CSVLogger

from keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split

from keras.optimizers import Adam

from keras.callbacks import LearningRateScheduler

from keras.losses import categorical_crossentropy

from keras.optimizers import Adam

#import segmentation_models as sm

from albumentations import*

import cv2

from random import shuffle                            #

import os

import matplotlib.pyplot as plt

from skimage.io import imsave





import time                                           # measuring training and test time

from glob import glob                                 # path control

import tqdm

from scipy.ndimage.morphology import binary_fill_holes

from skimage.morphology import remove_small_objects

from scipy.ndimage.filters import gaussian_filter

import skimage.morphology

from skimage import io, exposure, img_as_uint, img_as_float

from skimage.io import imsave, imread

from skimage.morphology import label

from skimage.morphology import watershed

from skimage.feature import peak_local_max

#import segmentation_models as sm

from scipy import ndimage as ndi
# Dice loss function

def dice_coef(y_true, y_pred):

    smooth = 1.

    y_true_f = K.flatten(y_true)

    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f)

    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)



def dice_loss(y_true, y_pred):

    return 1 - dice_coef(y_true, y_pred)

#####################################################################################

# Combination of Dice and binary cross entophy loss function

def bce_dice_loss(y_true, y_pred):

    return 0.5 * keras.losses.binary_crossentropy(y_true, y_pred) - dice_coef(y_true, y_pred)

########################################################################################

# custom callsback (decaying learning rate)

def step_decay_schedule(initial_lr=1e-3, decay_factor=0.75, epochs_drop=1000):

    '''

    Wrapper function to create a LearningRateScheduler with step decay schedule.

    '''

    def schedule(epoch):

        return initial_lr * (decay_factor ** np.floor(epoch/epochs_drop))

    

    return LearningRateScheduler(schedule, verbose = 1)

#######################################################################################################

def binary_unet( IMG_CHANNELS, LearnRate):

    inputs = Input((None, None, IMG_CHANNELS))

    s = Lambda(lambda x: x / 255) (inputs)



    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='glorot_uniform', padding='same') (s)

    c1 = Dropout(0.1) (c1)

    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='glorot_uniform', padding='same') (c1)

    p1 = MaxPooling2D((2, 2)) (c1)



    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='glorot_uniform', padding='same') (p1)

    c2 = Dropout(0.1) (c2)

    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='glorot_uniform', padding='same') (c2)

    p2 = MaxPooling2D((2, 2)) (c2)



    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='glorot_uniform', padding='same') (p2)

    c3 = Dropout(0.1) (c3)

    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='glorot_uniform', padding='same') (c3)

    p3 = MaxPooling2D((2, 2)) (c3)



    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='glorot_uniform', padding='same') (p3)

    c4 = Dropout(0.1) (c4)

    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='glorot_uniform', padding='same') (c4)

    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)



    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='glorot_uniform', padding='same') (p4)

    c5 = Dropout(0.1) (c5)

    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='glorot_uniform', padding='same') (c5)



    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c5)

    u6 = concatenate([u6, c4])

    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='glorot_uniform', padding='same') (u6)

    c6 = Dropout(0.1) (c6)

    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='glorot_uniform', padding='same') (c6)



    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c6)

    u7 = concatenate([u7, c3])

    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='glorot_uniform', padding='same') (u7)

    c7 = Dropout(0.1) (c7)

    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='glorot_uniform', padding='same') (c7)



    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c7)

    u8 = concatenate([u8, c2])

    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='glorot_uniform', padding='same') (u8)

    c8 = Dropout(0.1) (c8)

    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='glorot_uniform', padding='same') (c8)



    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c8)

    u9 = concatenate([u9, c1], axis=3)

    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='glorot_uniform', padding='same') (u9)

    c9 = Dropout(0.1) (c9)

    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='glorot_uniform', padding='same') (c9)



    outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9) # for binary



    model = Model(inputs=[inputs], outputs=[outputs])

    model.compile(optimizer = Adam(lr=LearnRate), loss= bce_dice_loss , metrics=[dice_coef]) #for binary



    #model.summary()

    return model

#######################################################################################################

def deeper_binary_unet(IMG_CHANNELS, LearnRate):

    # Build U-Net model

    inputs = Input((None, None, IMG_CHANNELS))

    #s = Lambda(lambda x: x / 255) (inputs)



    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (inputs)

    c1 = Dropout(0.1) (c1)

    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c1)

    p1 = MaxPooling2D((2, 2)) (c1)



    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p1)

    c2 = Dropout(0.1) (c2)

    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c2)

    p2 = MaxPooling2D((2, 2)) (c2)



    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p2)

    c3 = Dropout(0.1) (c3)

    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c3)

    p3 = MaxPooling2D((2, 2)) (c3)



    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p3)

    c4 = Dropout(0.1) (c4)

    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c4)

    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

    

    

    c4_new = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p4)

    c4_new = Dropout(0.1) (c4_new)

    c4_new = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c4_new)

    p4_new = MaxPooling2D(pool_size=(2, 2)) (c4_new)



    c5 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p4_new)

    c5 = Dropout(0.1) (c5)

    c5 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c5)

    

    

    u6_new = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same') (c5)

    u6_new = concatenate([u6_new, c4_new])

    c6_new = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u6_new)

    c6_new = Dropout(0.1) (c6_new)

    c6_new = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c6_new)



    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c6_new)

    u6 = concatenate([u6, c4])

    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u6)

    c6 = Dropout(0.1) (c6)

    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c6)



    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c6)

    u7 = concatenate([u7, c3])

    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u7)

    c7 = Dropout(0.1) (c7)

    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c7)



    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c7)

    u8 = concatenate([u8, c2])

    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u8)

    c8 = Dropout(0.1) (c8)

    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c8)



    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c8)

    u9 = concatenate([u9, c1], axis=3)

    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u9)

    c9 = Dropout(0.1) (c9)

    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c9)



    outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)



    model_deeper = Model(inputs=[inputs], outputs=[outputs])

    model_deeper.compile(optimizer = Adam(lr=LearnRate), loss= bce_dice_loss , metrics=[ dice_coef])

    model_deeper.summary()

    return model_deeper
# augmentation function

def albumentation_aug(p=1.0, crop_size_row = 448, crop_size_col = 448 ):

    return Compose([

        RandomCrop(crop_size_row, crop_size_col, always_apply=True, p=1),

        CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), always_apply=False, p=0.5),

        RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, brightness_by_max=True, p=0.4),

        HueSaturationValue(hue_shift_limit=20, sat_shift_limit=20, val_shift_limit=20, p=0.1),

        HorizontalFlip(always_apply=False, p=0.5),

        VerticalFlip(always_apply=False, p=0.5),

        RandomRotate90(always_apply=False, p=0.5),

        #ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=20, interpolation=1, border_mode=4, value=None, mask_value=None, always_apply=False, p=0.3),

    ], p=p) # --> this p has the second proiroty comapred to the p inside each argument (e.g. HorizontalFlip(always_apply=False, p=0.5) )

###########################################################

def albumentation_aug_light(p=1.0, crop_size_row = 448, crop_size_col = 448):

    return Compose([

        RandomCrop(crop_size_row, crop_size_col, always_apply=True, p=1.0),

        HorizontalFlip(always_apply=False, p=0.5),

        VerticalFlip(always_apply=False, p=0.5),

        RandomRotate90(always_apply=False, p=0.5),

        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=20, interpolation=1, border_mode=4, value=None, mask_value=None, always_apply=False, p=0.1),

    ], p=p, additional_targets={'mask1': 'mask','mask2': 'mask'}) # --> this p has the second proiroty comapred to the p inside each argument (e.g. HorizontalFlip(always_apply=False, p=0.5) )

def get_dice_1(true, pred):

    """

        Traditional dice

    """

    # cast to binary 1st

    true = np.copy(true)

    pred = np.copy(pred)

    true[true > 0] = 1

    pred[pred > 0] = 1

    inter = true * pred

    denom = true + pred

    return 2.0 * np.sum(inter) / np.sum(denom)

##############################################################################################

def get_fast_aji(true, pred):

    """

    AJI version distributed by MoNuSeg, has no permutation problem but suffered from 

    over-penalisation similar to DICE2

    Fast computation requires instance IDs are in contiguous orderding i.e [1, 2, 3, 4] 

    not [2, 3, 6, 10]. Please call `remap_label` before hand and `by_size` flag has no 

    effect on the result.

    """

    true = np.copy(true) # ? do we need this

    pred = np.copy(pred)

    true_id_list = list(np.unique(true))

    pred_id_list = list(np.unique(pred))



    true_masks = [None,]

    for t in true_id_list[1:]:

        t_mask = np.array(true == t, np.uint8)

        true_masks.append(t_mask)

    

    pred_masks = [None,]

    for p in pred_id_list[1:]:

        p_mask = np.array(pred == p, np.uint8)

        pred_masks.append(p_mask)

    

    # prefill with value

    pairwise_inter = np.zeros([len(true_id_list) -1, 

                               len(pred_id_list) -1], dtype=np.float64)

    pairwise_union = np.zeros([len(true_id_list) -1, 

                               len(pred_id_list) -1], dtype=np.float64)



    # caching pairwise

    for true_id in true_id_list[1:]: # 0-th is background

        t_mask = true_masks[true_id]

        pred_true_overlap = pred[t_mask > 0]

        pred_true_overlap_id = np.unique(pred_true_overlap)

        pred_true_overlap_id = list(pred_true_overlap_id)

        for pred_id in pred_true_overlap_id:

            if pred_id == 0: # ignore

                continue # overlaping background

            p_mask = pred_masks[pred_id]

            total = (t_mask + p_mask).sum()

            inter = (t_mask * p_mask).sum()

            pairwise_inter[true_id-1, pred_id-1] = inter

            pairwise_union[true_id-1, pred_id-1] = total - inter

    #

    pairwise_iou = pairwise_inter / (pairwise_union + 1.0e-6)

    # pair of pred that give highest iou for each true, dont care 

    # about reusing pred instance multiple times

    paired_pred = np.argmax(pairwise_iou, axis=1)

    pairwise_iou = np.max(pairwise_iou, axis=1)

    # exlude those dont have intersection

    paired_true = np.nonzero(pairwise_iou > 0.0)[0]

    paired_pred = paired_pred[paired_true]

    # print(paired_true.shape, paired_pred.shape)

    overall_inter = (pairwise_inter[paired_true, paired_pred]).sum()

    overall_union = (pairwise_union[paired_true, paired_pred]).sum()

    #

    paired_true = (list(paired_true + 1)) # index to instance ID

    paired_pred = (list(paired_pred + 1))

    # add all unpaired GT and Prediction into the union

    unpaired_true = np.array([idx for idx in true_id_list[1:] if idx not in paired_true])

    unpaired_pred = np.array([idx for idx in pred_id_list[1:] if idx not in paired_pred])

    for true_id in unpaired_true:

        overall_union += true_masks[true_id].sum()

    for pred_id in unpaired_pred:

        overall_union += pred_masks[pred_id].sum()

    #

    aji_score = overall_inter / overall_union

    return aji_score

##############################################################################################

def remap_label(pred, by_size=False):

    """

    Rename all instance id so that the id is contiguous i.e [0, 1, 2, 3] 

    not [0, 2, 4, 6]. The ordering of instances (which one comes first) 

    is preserved unless by_size=True, then the instances will be reordered

    so that bigger nucler has smaller ID

    Args:

        pred    : the 2d array contain instances where each instances is marked

                  by non-zero integer

        by_size : renaming with larger nuclei has smaller id (on-top)

    """

    pred_id = list(np.unique(pred))

    pred_id.remove(0)

    if len(pred_id) == 0:

        return pred # no label

    if by_size:

        pred_size = []

        for inst_id in pred_id:

            size = (pred == inst_id).sum()

            pred_size.append(size)

        # sort the id by size in descending order

        pair_list = zip(pred_id, pred_size)

        pair_list = sorted(pair_list, key=lambda x: x[1], reverse=True)

        pred_id, pred_size = zip(*pair_list)



    new_pred = np.zeros(pred.shape, np.int32)

    for idx, inst_id in enumerate(pred_id):

        new_pred[pred == inst_id] = idx + 1    

    return new_pred
# other useful finction for training



def get_id_from_file_path(file_path, indicator):

    return file_path.split(os.path.sep)[-1].replace(indicator, '')

############################################################

def chunker(seq, seq2, size):

    return ([seq[pos:pos + size], seq2[pos:pos + size]] for pos in range(0, len(seq), size))

############################################################

def data_gen_heavy(list_files, list_files2, batch_size, p , size_row, size_col, distance_unet_flag = 0, augment=False, BACKBONE_model = 'efficientnetb0', use_pretrain_flag =1):

    #preprocess_input = sm.get_preprocessing(BACKBONE_model)

    crop_size_row = size_row

    crop_size_col = size_col

    aug = albumentation_aug(p, crop_size_row, crop_size_col)



    while True:

        #shuffle(list_files)

        for batch in chunker(list_files,list_files2, batch_size):

            #X = [cv2.resize(cv2.imread(x), (size, size)) for x in batch]

            X = []

            Y = []



            for count in range(len(batch[0])):

                # x = cv2.resize(cv2.imread(batch[0][count]), (size_col, size_row))

                # x_mask = cv2.resize(cv2.imread(batch[1][count], cv2.IMREAD_GRAYSCALE), (size_col, size_row))

                x = cv2.imread(batch[0][count])

                x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)

                x_mask = cv2.imread(batch[1][count], cv2.IMREAD_GRAYSCALE)

                

                x_mask_temp = np.zeros((x_mask.shape[0], x_mask.shape[1]))

                x_mask_temp[x_mask == 255] = 1

                



                if distance_unet_flag == False:

                    if augment:

                        augmented = aug(image= x, mask= x_mask_temp)

                        x = augmented['image']

                        if use_pretrain_flag == 1:

                            x = preprocess_input(x)

                        x_mask_temp = augmented['mask']

                    X.append(x)

                    Y.append(x_mask_temp)

                    #imsave('/media/masih/wd/projects/MoNuSAC_binary/results/images/an/{}_binary.png'.format(get_id_from_file_path(batch[0][count], '.png')), x_mask_epithelial)

                    #imsave('/media/masih/wd/projects/MoNuSAC_binary/results/images/an/{}.png'.format(get_id_from_file_path(batch[0][count], '.tif')), x)

                else:

                    if augment:

                        augmented = aug(image=x, mask=x_mask)

                        x = augmented['image']

                        if use_pretrain_flag == 1:

                            x = preprocess_input(x)

                        x_mask = augmented['mask']



                    X.append(x)

                    x_mask = (x_mask - np.min(x_mask))/ (np.max(x_mask) - np.min(x_mask) + 0.0000001)

                    Y.append(x_mask)



                del x_mask

                del x_mask_temp

                del x

            Y = np.expand_dims(np.array(Y), axis=3)

            Y = np.array(Y)

            yield np.array(X), np.array(Y)

# create folders to save the best models and images (if needed) for each fold

if not os.path.exists('/kaggle/working/images/'):

    os.makedirs('/kaggle/working/images/')

if not os.path.exists('/kaggle/working/models/'):

    os.makedirs('/kaggle/working/models/')    

if not os.path.exists(opts['results_save_path']+ 'stage1/validation/'):

    os.makedirs(opts['results_save_path'] + 'stage1/validation/')

if not os.path.exists(opts['results_save_path']+ 'stage2/validation/'):

    os.makedirs(opts['results_save_path'] + 'stage2/validation/')

if not os.path.exists(opts['results_save_path']+ 'stage2/validation/figure/'):

    os.makedirs(opts['results_save_path'] + 'stage2/validation/figure/')
train_files = glob('{}*{}'.format(opts['train_dir'], opts['imageType_train']))

train_files_mask = glob('{}*.png'.format(opts['train_label_dir']))

train_files_dis = glob('{}*.png'.format(opts['train_dis_dir']))

train_files_labels = glob('{}*.tif'.format(opts['train_label_masks']))





train_files.sort()

train_files_mask.sort()

train_files_dis.sort()

train_files_labels.sort()

print("Total number of training images:", len(train_files))
# we have 10 organ in this dataset

train_files
# creating 10 folds to perfrom 10 fold cross-validation (for each fold images from the 9 organs are used for training and the images from one organ are used as validation)



for k in range(opts['k_fold']):

    if k ==0:

        fold1 = train_files[0: int(np.round(len(train_files) / opts['k_fold']))]

    else:

        globals()["fold" + str(k + 1)] = train_files[int(np.round(len (train_files) / opts['k_fold']) * k): int(np.round(len(train_files) / opts['k_fold']) * (k+1))]

print("length of each fold:", len(fold1))



# for binary mask

for k in range(opts['k_fold']):

    if k ==0:

        fold_mask1 = train_files_mask[0: int(np.round(len(train_files_mask) / opts['k_fold']))]

    else:

        globals()["fold_mask" + str(k + 1)] = train_files_mask[int(np.round(len (train_files_mask) / opts['k_fold']) * k): int(np.round(len(train_files_mask) / opts['k_fold']) * (k+1))]



# for distance mask

for k in range(opts['k_fold']):

    if k ==0:

        fold_dis1 = train_files_dis[0: int(np.round(len(train_files_dis) / opts['k_fold']))]

    else:

        globals()["fold_dis" + str(k + 1)] = train_files_dis[int(np.round(len (train_files_dis) / opts['k_fold']) * k): int(np.round(len(train_files_dis) / opts['k_fold']) * (k+1))]



# for label masks (just for evaluation)

for k in range(opts['k_fold']):

    if k ==0:

        fold_label1 = train_files_labels[0: int(np.round(len(train_files_labels) / opts['k_fold']))]

    else:

        globals()["fold_label" + str(k + 1)] = train_files_labels[int(np.round(len (train_files_labels) / opts['k_fold']) * k): int(np.round(len(train_files_labels) / opts['k_fold']) * (k+1))]

# main training loop (for all 10 fold cross-validation)

start_time = time.time()

dice_pure_unet = np.zeros([opts['k_fold'],len(fold1)])

AJI_pure_unet = np.zeros([opts['k_fold'],len(fold1)])



dice_unet_watershed = np.zeros([opts['k_fold'],len(fold1)])

AJI_unet_watershed = np.zeros([opts['k_fold'],len(fold1)])





for K_fold in range(opts['k_fold']):    

    train = []

    train_mask = []

    train_dis = []

    

    val = eval('fold' + str(K_fold + 1))

    val_mask = eval('fold_mask' + str(K_fold + 1))

    val_dis = eval('fold_dis' + str(K_fold + 1))

    val_label = eval('fold_label' + str(K_fold + 1))



    for ii in range(opts['k_fold']):

        if ii != K_fold:

            train = eval('fold' + str(ii + 1)) + train



    for ii in range(opts['k_fold']):

        if ii != K_fold:

            train_mask = eval('fold_mask' + str(ii + 1)) + train_mask



    for ii in range(opts['k_fold']):

        if ii != K_fold:

            train_dis = eval('fold_dis' + str(ii + 1)) + train_dis



    if opts['k_fold'] == 1: # for no cross validation the training will be with all training images

        train = train_files

        train_mask = train_files_mask

        train_dis = train_files_dis

   

    random.Random(opts['random_seed_num']).shuffle(train)

    random.Random(opts['random_seed_num']).shuffle(train_mask)

    random.Random(opts['random_seed_num']).shuffle(train_dis)

 



    ## creating validation data for each fold (just for evaluation)

    # it is not included in the main training loop for a faster training

    validation_X = []

    validation_Y = []

    validation_DIS = []

    if len(val)<200: # memory consideration

        for an in range(len(val)):

            x = cv2.imread(val[an])

            x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)



            aug = albumentation_aug_light(1, opts['crop_size'], opts['crop_size'])

            #augmented = aug(image=x)

            #x = augmented['image']

            if opts['use_pretrained_flag'] == 1:

                x = preprocess_input(x)

            img_mask = imread(val_label[an])

            validation_X.append(x)

            validation_Y.append(img_mask)



    else:

        for an in range(200):

            x = cv2.imread(val[an])

            x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)



            aug = albumentation_aug_light(1, opts['crop_size'], opts['crop_size'])

            #augmented = aug(image=x)

            #x = augmented['image']

            if opts['use_pretrained_flag'] ==1:

                x = preprocess_input(x)

            img_mask = imread(val_label[an])

            validation_X.append(x)

            validation_Y.append(img_mask)



    validation_X = np.array(validation_X)

    validation_Y = np.array(validation_Y)

    

    

    model_path = opts['models_save_path'] + 'raw_unet_{}.h5'.format(K_fold+1)

    logger = CSVLogger(opts['models_save_path']+ 'raw_unet_{}.log'.format(K_fold + 1))

    LR_drop = step_decay_schedule(initial_lr= opts['init_LR'], decay_factor = opts['LR_decay_factor'], epochs_drop = opts['LR_drop_after_nth_epoch'])

    model_raw = deeper_binary_unet(opts['number_of_channel'], opts['init_LR'])

    checkpoint = ModelCheckpoint(model_path, monitor='val_dice_coef', verbose=1, save_best_only=True, mode='max', save_weights_only = True)

    

    # training

    history = model_raw.fit_generator(data_gen_heavy(train,

                                                     train_mask,

                                                     opts['batch_size'],

                                                     1,

                                                     opts['crop_size'], opts['crop_size'],

                                                     distance_unet_flag=0,

                                                     augment=True,

                                                     BACKBONE_model=opts['pretrained_model'],

                                                     use_pretrain_flag=opts['use_pretrained_flag']),

                                      validation_data=data_gen_heavy(val,

                                                                     val_mask,

                                                                     opts['batch_size'],

                                                                     1,

                                                                     opts['crop_size'], opts['crop_size'],

                                                                     distance_unet_flag=0,

                                                                     augment=True,

                                                                     BACKBONE_model=opts['pretrained_model'],

                                                                     use_pretrain_flag=opts['use_pretrained_flag']),

                                      validation_steps=1,

                                      epochs=opts['epoch_num_stage1'], verbose=1,

                                      callbacks=[checkpoint, logger, LR_drop],

                                      steps_per_epoch=(len(train) // opts['batch_size']) // opts['quick_run'])

    

    model_raw.load_weights(opts['models_save_path'] + 'raw_unet_{}.h5'.format(K_fold + 1))



    ## predication on validation set

    preds_val = model_raw.predict(validation_X, verbose=1, batch_size=1)

    preds_val_t = (preds_val > opts['treshold']).astype(np.uint8)





    for val_len in range(len(preds_val)):

        # with watershed post processing

        local_maxi = peak_local_max(np.squeeze(preds_val[val_len]), indices=False,exclude_border=False, footprint=np.ones((15, 15)))

        markers = ndi.label(local_maxi)[0]

        labels = watershed(-np.squeeze(preds_val[val_len]), markers,mask = np.squeeze(preds_val_t[[val_len]]))

        labels[np.squeeze(preds_val_t[[val_len]])==0] = 0

        

        # without post processing 

        pred = np.squeeze(preds_val_t[val_len])

        label_pred = label(pred)

        

        label_pred = remap_label(label_pred)

        validation_Y[val_len] = remap_label(validation_Y[val_len])

        labels = remap_label(labels)

        

        

        

        dice_pure_unet[K_fold, val_len]= get_dice_1(  label_pred,   validation_Y[val_len])

        AJI_pure_unet[K_fold, val_len] = get_fast_aji(label_pred, validation_Y[val_len])

        

        dice_unet_watershed[K_fold, val_len]= get_dice_1(  labels,   validation_Y[val_len])

        AJI_unet_watershed[K_fold, val_len] = get_fast_aji(labels, validation_Y[val_len])

        

        

    print('==========')    

    print('average dice pure Unet for fold{}:'.format(K_fold), np.mean(dice_pure_unet[K_fold, :]))

    print('average AJI pure Unet for fold{}:'.format(K_fold), np.mean(AJI_pure_unet[K_fold, :]))

    print('==========') 

    

    print('==========')    

    print('average Dice Unet watershed for fold{}:'.format(K_fold), np.mean(dice_unet_watershed[K_fold, :]))

    print('average AJI Unet watershed for fold{}:'.format(K_fold), np.mean(AJI_unet_watershed[K_fold, :]))

    print('==========') 

finish_time = time.time() 

print('==========') 

print('total training time (all 10 folds):',  (finish_time- start_time)/60, 'minutes')

import pandas as pd

organ_name = ['Human_AdrenalGland', 'Human_Larynx', 'Human_LymphNodes', 'Human_Mediastinum', 

              'Human_Pancreas','Human_Pleura', 'Human_Skin', 'Human_Testes' , 'Human_Thymus', 'Human_ThyroidGland']

df = pd.DataFrame({'Oragn': organ_name, 'DICE mean': np.mean(dice_pure_unet, axis = 1), 'AJI mean': np.mean(AJI_pure_unet, axis = 1)}) 

df.to_csv('final_scores_pure_unet.csv', index=False)

print('averge overall dice score (pure Unet):',"{:.2f}".format(np.mean(dice_pure_unet)*100), '%')

print('averge overall AJI score (pure Unet):', "{:.2f}".format(np.mean(AJI_pure_unet)*100), '%')

df
import pandas as pd

organ_name = ['Human_AdrenalGland', 'Human_Larynx', 'Human_LymphNodes', 'Human_Mediastinum', 

              'Human_Pancreas','Human_Pleura', 'Human_Skin', 'Human_Testes' , 'Human_Thymus', 'Human_ThyroidGland']

df = pd.DataFrame({'Oragn': organ_name, 'DICE mean': np.mean(dice_unet_watershed, axis = 1), 'AJI mean': np.mean(AJI_unet_watershed, axis = 1)}) 

df.to_csv('final_scores_unet_watershed.csv', index=False)

print('averge overall dice score (Unet + watershed):', "{:.2f}".format(np.mean(dice_unet_watershed)*100),'%')

print('averge overall AJI score (Unet + watershed):', "{:.2f}".format(np.mean(AJI_unet_watershed)*100),'%')

df