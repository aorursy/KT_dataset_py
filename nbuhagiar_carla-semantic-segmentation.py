# General Data Science

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



# Machine Learning

from fastai.vision.data import SegmentationItemList, imagenet_stats

from fastai.vision.transform import get_transforms

from fastai.vision.learner import unet_learner

from fastai.vision.models import resnet34



# Miscellaneous

import os

from pathlib import Path

from shutil import copyfile, rmtree
img_path = Path("/kaggle/working/img")

labels_path = Path("/kaggle/working/labels")

os.mkdir(img_path)

os.mkdir(labels_path)



for dirname, _, filenames in os.walk('/kaggle/input'):

    if dirname.endswith("CameraRGB") and any(dataset in dirname for dataset in ["dataa", "datab", "datac", "datad", "datae"]):

        for filename in filenames:

            copyfile(os.path.join(dirname, filename), img_path/filename)

    elif dirname.endswith("CameraSeg") and any(dataset in dirname for dataset in ["dataa", "datab", "datac", "datad", "datae"]):

        for filename in filenames:

            copyfile(os.path.join(dirname, filename), labels_path/filename)
codes = np.array(["Unlabeled",

                  "Building",

                  "Fence",

                  "Other",

                  "Pedestrian",

                  "Pole",

                  "Road line",

                  "Road",

                  "Sidewalk",

                  "Vegetation",

                  "Car",

                  "Wall",

                  "Traffic sign",])
data = (SegmentationItemList.from_folder(img_path)

        .split_by_rand_pct(seed=0)

        .label_from_func(lambda x: labels_path/f"{x.stem}{x.suffix}", classes=codes)

        .transform(get_transforms(), tfm_y=True)

        .databunch(bs=4)

        .normalize(imagenet_stats))

data
data.show_batch(2)
learn = unet_learner(data, resnet34, model_dir="/kaggle/working")
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(3, slice(7e-4))
learn.show_results(3)
rmtree(img_path)

rmtree(labels_path)