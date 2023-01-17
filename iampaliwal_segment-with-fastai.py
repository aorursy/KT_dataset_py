import numpy as np

import pandas as pd 

from fastai.vision import *

from fastai import *

import zipfile

import torch.nn as nn

import os

# print(os.listdir("../input/carvana-image-masking-challenge"))
def unzip_file(path_name):

    with zipfile.ZipFile(path_name, 'r') as zip_ref:

        zip_ref.extractall("/kaggle/working/")
unzip_file("/kaggle/input/carvana-image-masking-challenge/train_masks.zip")

unzip_file("/kaggle/input/carvana-image-masking-challenge/train.zip")
from subprocess import check_output

print(check_output(["ls", "/kaggle/working/"]).decode("utf8"))
path = Path("/kaggle/working/")
image_path = path/"train"

mask_path = path/"train_masks"
img = open_image("/kaggle/input/carvana-image-masking-challenge/29bb3ece3180_11.jpg")

img.show(figsize = (5,5))

img.shape
images = get_image_files("/kaggle/working/train")

images.sort()

images[:5]
masks = get_image_files("/kaggle/working/train_masks")

masks.sort()

masks[:4]
get_masks = lambda x: mask_path/f'{x.stem}_mask.gif'
msk = open_mask(get_masks(images[1]), div = True)

msk.show(figsize = (10,7))
class LabelList(SegmentationLabelList):

    def open(self,fn): 

        return open_mask(fn, div=True)

    

class ItemList(ImageList):

    _label_cls= LabelList
#Defining the dataset

src_size = np.array(msk.shape[1:])

size = src_size//4

codes = ["background", "car"]

# size = 224

bs = 4

src = (ItemList.from_folder(image_path)

       .use_partial_data(sample_pct=0.1)

       .split_by_rand_pct(0.2,42)

       .label_from_func(get_masks, classes = codes))
data = (src.transform(get_transforms(), size = size, tfm_y = True)

        .databunch(bs=bs)

        .normalize(imagenet_stats))
data.show_batch(2,figsize = (7,7))
data.show_batch(2,figsize = (7,7), ds_type = DatasetType.Valid)
!cp '/kaggle/input/pretrained-pytorch-models/resnet50-19c8e357.pth' '/root/.cache/torch/checkpoints/resnet50-19c8e357.pth'
wd = 1e-2

learn = unet_learner(data, models.resnet50, metrics = [dice], wd = wd).to_fp16()
learn.lr_find()

learn.recorder.plot()
lr = 1e-03

learn.fit_one_cycle(3,slice(lr), pct_start = 0.9)
learn.save('stage-1')
learn.load('stage-1')
learn.unfreeze()
# common practice

lrs = slice(lr/400, lr/4)
learn.fit_one_cycle(5, lrs, pct_start = 0.8)