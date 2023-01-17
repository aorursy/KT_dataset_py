import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

from os import listdir

from PIL import Image, ImageDraw, ImageFont

import matplotlib.pyplot as plt

%matplotlib inline

import cv2

print(os.listdir("../input"))



train_df=pd.read_csv("../input/train.csv")
train_df.head()
train_df.info()
train_df_nan=train_df[train_df['labels'].isna()]
train_df_nan.count()
img=Image.open("../input/train_images/100241706_00004_2.jpg")

plt.imshow(img)
translation_df=pd.read_csv("../input/unicode_translation.csv")
translation_df.head(10)