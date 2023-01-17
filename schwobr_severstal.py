%load_ext autoreload

%autoreload 2



import gc

import cv2

import PIL

import random

import numpy as np

import os

import pydicom

import matplotlib.pyplot as plt

import pandas as pd

from tqdm import tqdm_notebook as tqdm

import pdb

from dataclasses import dataclass

from functools import partial

import datetime

from io import BytesIO

from enum import IntEnum

import sys

import re



from skimage.morphology import label

from sklearn.model_selection import KFold, GridSearchCV



from torchvision import transforms

import torchvision.transforms.functional as TF

from torch.nn.functional import binary_cross_entropy_with_logits

import torch

import torch.nn as nn

from torch.utils.data import WeightedRandomSampler, Sampler, DataLoader

from torch.optim import SGD



from fastai.vision.data import SegmentationItemList, SegmentationLabelList, ImageList, imagenet_stats

from fastai.data_block import FloatList, FloatItem, ItemList, PreProcessor, LabelList, LabelLists, EmptyLabelList, MixedItemList

from fastai.basic_data import DatasetType, DataBunch, DeviceDataLoader

from fastai.basic_train import Learner

from fastai.train import GradientClipping

from fastai.callback import OptimWrapper

from fastai.core import *

from fastai.vision.image import Image, ImageSegment, image2np, pil2tensor, rle_decode, rle_encode, open_image, open_mask_rle

from fastai.vision.transform import get_transforms

from fastai.vision.learner import unet_learner, cnn_learner

import fastai.vision.models as mod

from fastai.callbacks import SaveModelCallback, LearnerCallback

from fastai.callbacks.tensorboard import LearnerTensorboardWriter

from fastai.metrics import accuracy



from pathlib import Path



# IMAGE SIZES

TRAIN_SIZE = (64, 400)

MAX_SIZE = 1388

TEST_SIZE = 256

TEST_OVERLAP = 64

IMG_CHANNELS = 3



# PATHS

PROJECT_PATH = Path(

    '/kaggle/working')

DATA = Path('/kaggle/input/severstal-steel-defect-detection')

OLD_MODEL = Path('/kaggle/input/severstal/models')

TRAIN_PATH = DATA/'train_images'

TEST_PATH = DATA/'test_images'

MODELS_PATH = PROJECT_PATH/'models/'

SUB_PATH = PROJECT_PATH/'submissions/'

NEW_DATA = PROJECT_PATH/'data'

LABELS_OLD = DATA/'train.csv'

LABELS = NEW_DATA/'train_new.csv'



# LEARNER CONFIG

BATCH_SIZE = 8

WD = 1e-3

LR = 2e-4

GROUP_LIMITS = None

FREEZE_UNTIL = None

EPOCHS = 10

UNFROZE_EPOCHS = 10

PRETRAINED = True

MODEL = 'resnet34'

CLASSES = ['pneum']
def add_test(self, items, label=None, tfms=None, tfm_y=None):

    "Add the `items` as a test set. Pass along `label` otherwise label them with `EmptyLabel`."

    self.label_list.add_test(items, label=label, tfms=tfms, tfm_y=tfm_y)

    vdl = self.valid_dl

    dl = DataLoader(self.label_list.test, vdl.batch_size, shuffle=False, drop_last=False, num_workers=vdl.num_workers)

    self.test_dl = DeviceDataLoader(dl, vdl.device, vdl.tfms, vdl.collate_fn)

DataBunch.add_test = add_test
def add_test(self, items, label=None, tfms=None, tfm_y=None):

    "Add test set containing `items` with an arbitrary `label`."

    # if no label passed, use label of first training item

    if label is None: labels = EmptyLabelList([0] * len(items))

    else: labels = self.valid.y.new([label] * len(items)).process()

    if isinstance(items, MixedItemList): items = self.valid.x.new(items.item_lists, inner_df=items.inner_df).process()

    elif isinstance(items, ItemList): items = self.valid.x.new(items.items, inner_df=items.inner_df).process()

    else: items = self.valid.x.new(items).process()

    self.test = self.valid.new(items, labels, tfms=tfms, tfm_y=tfm_y)

    return self



def add_test_folder(self, test_folder='test', label=None, tfms=None, tfm_y=None):

    "Add test set containing items from `test_folder` and an arbitrary `label`."

    # note: labels will be ignored if available in the test dataset

    items = self.x.__class__.from_folder(self.path/test_folder)

    return self.add_test(items.items, label=label, tfms=tfms, tfm_y=tfm_y)



LabelLists.add_test = add_test

LabelLists.add_test_folder = add_test_folder
if not MODELS_PATH.is_dir():

    MODELS_PATH.mkdir()

if not SUB_PATH.is_dir():

    SUB_PATH.mkdir()

if not NEW_DATA.is_dir():

    NEW_DATA.mkdir()
@classmethod

def from_csv(cls, path, csv_name, header='infer', **kwargs):

    "Get the filenames in `path/csv_name` opened with `header`."

    path = Path(path)

    df = pd.read_csv(csv_name, header=header)

    return cls.from_df(df, path=path, **kwargs)

ImageList.from_csv = from_csv
def change_csv(old, new):

    df = pd.read_csv(old)



    def group_func(df, i):

        reg = re.compile(r'(.+)_\d$')

        return reg.search(df['ImageId_ClassId'].loc[i]).group(1)



    group = df.groupby(lambda i: group_func(df, i))



    df = group.agg({'EncodedPixels': lambda x: list(x)})



    df['ImageId'] = df.index

    df = df.reset_index(drop=True)



    df[[f'EncodedPixels_{k}' for k in range(1, 5)]] = pd.DataFrame(df['EncodedPixels'].values.tolist())

    

    df = df.drop(columns='EncodedPixels')

    df = df.fillna(value=' ')

    df.to_csv(new, index=False)

    return df
if not LABELS.is_file():

    change_csv(LABELS_OLD, LABELS).head()
def show_image(img, ax=None, figsize=(3,3), hide_axis=True, cmap='binary',

                alpha=None, **kwargs):

    "Display `Image` in notebook."

    if ax is None: fig,ax = plt.subplots(figsize=figsize)

    img = img.data.float()

    for k in range(img.size(0)):

        img[k] *= (k+1)

    img = img.sum(0).unsqueeze(0)

    ax.imshow(image2np(img.data), cmap=cmap, alpha=alpha, **kwargs)

    if hide_axis: ax.axis('off')

    return ax
class Mask(ImageSegment):

    def show(self, ax=None, figsize=(3,3), title=None, hide_axis=True,

        cmap='tab20', alpha=0.5, **kwargs):

        "Show the `ImageSegment` on `ax`."

        ax = show_image(self, ax=ax, hide_axis=hide_axis, cmap=cmap, figsize=figsize,

                        interpolation='nearest', alpha=alpha, vmin=0, **kwargs)

        if title: ax.set_title(title)    
class MultiClassSegList(SegmentationLabelList):

    def open(self, id_rles):

        image_id, rles = id_rles[0], id_rles[1:]

        shape = open_image(self.path/image_id).shape[-2:]       

        final_mask = torch.zeros((1, *shape))

        for k, rle in enumerate(rles):

            if isinstance(rle, str):

                mask = open_mask_rle(rle, shape).px.permute(0, 2, 1)

                final_mask += (k+1)*mask

        return ImageSegment(final_mask)
def load_data(path, csv, bs=32, size=(128, 800)):

    train_list = (SegmentationItemList.

                  from_csv(path, csv).

                  split_by_rand_pct(valid_pct=0.2).

                  label_from_df(cols=list(range(5)), label_cls=MultiClassSegList, classes=[0, 1, 2, 3, 4]).

                  transform(get_transforms(), size=size, tfm_y=True).

                  databunch(bs=bs, num_workers=0).

                  normalize(imagenet_stats))

    return train_list
db = load_data(TRAIN_PATH, LABELS, bs=BATCH_SIZE, size=TRAIN_SIZE)
db.show_batch(rows=1, figsize=(10, 10))
def dice(input, target, smooth=1., reduction='mean', smooth_num=True, **kwargs):

    iflat = nn.Softmax(dim=1)(input)[:, 1:].view(input.size(0), input.size(1), -1).float()

    one_hot = torch.zeros_like(input)

    one_hot.scatter_(1, target, 1)

    tflat = one_hot[:, 1:].view(one_hot.size(0), one_hot.size(1), -1).float()

    intersection = (iflat * tflat).sum(-1).mean(-1)

    smooth_u, corr = (smooth, 0) if smooth_num else (0, (1-tflat.max(-1).values))

    dice = corr + (2. * intersection + smooth_u)/((iflat + tflat).sum(-1).mean(-1) + smooth)

    if reduction=='mean':

        return dice.mean()

    elif reduction=='sum':

        return dice.sum()

    else:

        return dice
learner = unet_learner(db, mod.resnet50, wd=WD, pretrained=True, model_dir=MODELS_PATH, metrics=[dice])
learner.fit_one_cycle(15, slice(1e-3))
learner.unfreeze()
learner.fit_one_cycle(5, slice(1e-5))
learner.save('model')