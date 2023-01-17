# the fast.ai library

from fastai import *

from fastai.vision import *



# to inspect the directory

import os

from pathlib import Path



# for data manipulation

import pandas as pd



# for numerical analysis

import numpy as np
INPUT = Path("../input/understanding_cloud_organization")

os.listdir(INPUT)
TRAIN = INPUT/"test_images"

TEST = INPUT/"train_images"



print(os.listdir(TRAIN)[:5])

print(os.listdir(TEST)[:5])
train_df = pd.read_csv(INPUT/"train.csv")

train_df[["ImageId", "ClassId"]] = train_df["Image_Label"].str.split("_", expand=True)

train_df.drop(columns="Image_Label", inplace=True)

train_df = train_df[["ImageId", "ClassId", "EncodedPixels"]]

train_df = train_df.pivot(index="ImageId", columns="ClassId", values="EncodedPixels")

train_df.columns.name = ""

train_df = train_df.reset_index(level=0)

train_df.head()
mask = open_mask_rle(train_df.iloc[101]["Flower"], shape=(1400, 2100))

mask.show(figsize=(5,5), alpha=1)