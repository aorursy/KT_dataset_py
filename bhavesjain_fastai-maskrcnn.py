import numpy as np
import pandas as pd
from pathlib import Path
from fastai.vision import *
from fastai.callbacks.hooks import *
from fastai.utils.mem import *
# from progressbar import ProgressBar
import cv2
import os
import json
print(os.listdir("../input/fastai-factory-approach-using-mask-images/models/"))
path = Path("../input")
path_img = path/'train'
path_lbl = Path("../input/labels-impl")

# only the 27 apparel items, plus 1 for background
# model image size 224x224
category_num = 27 + 1
size = 224

# get and show categories
with open(path_lbl/"label_descriptions.json") as f:
    label_descriptions = json.load(f)

label_names = [x['name'] for x in label_descriptions['categories']]
print(label_names)

# I create an accuracy metric which excludes the background pixels
# not sure if this is correct
def acc_fashion(input, target):
    target = target.squeeze(1)
    mask = target != category_num - 1
    return (input.argmax(dim=1)==target).float().mean()
wd = 1e-2

# src = (ImageList.from_folder("../input/fashion/chris evans/",size=224).split_none())
# #transform(get_transforms(),size=224).
# data = (src.transform(size=300).label_empty().databunch(bs=16))#.normalize(imagenet_stats))
# data.show_batch(rows=3,figsize=(12,9))
data = (SegmentationItemList.from_folder('../input/fashion/chris hemsworth')
                            .split_none()
                            .label_empty()
                            .transform(size=256)
                            .databunch(bs=16)
                            .normalize(imagenet_stats))
data.show_batch(rows=3,figsize=(12,9))
learn = load_learner(data, models.resnet34).load('../input/fastai-factory-approach-using-mask-images/models')
model = unet_learner(data, models.resnet34, metrics=acc_fashion, wd=wd, model_dir="../input/fastai-factory-approach-using-mask-images/models/")
