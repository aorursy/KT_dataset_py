from PIL import Image, ImageDraw

import glob 

import json

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns

%matplotlib inline
train_df = pd.read_csv("../input/imaterialist-fashion-2020-fgvc7/train.csv")
train_df.keys()
train_df.loc[:30]
Unique_ImageId = set(train_df["ImageId"])
print(f"There are {len(train_df)} unique record in train.csv." )

print(f"There are {len(Unique_ImageId)} unique data." )
data_path = [ "../input/imaterialist-fashion-2020-fgvc7/train/" + Id + ".jpg" for Id in Unique_ImageId]
fig = plt.figure(figsize=(25, 16))

for i,im_path in enumerate(data_path[:16]):

    ax = fig.add_subplot(4, 4, i+1, xticks=[], yticks=[])

    im = Image.open(im_path)

    im = im.resize((350,480))

    plt.imshow(im)
fig = plt.figure(figsize=(25, 16))

for i,im_path in enumerate(data_path[32:48]):

    ax = fig.add_subplot(4, 4, i+1, xticks=[], yticks=[])

    im = Image.open(im_path)

    im = im.resize((350,480))

    plt.imshow(im)
fig = plt.figure(figsize=(25, 16))

for i,im_path in enumerate(data_path[16:32]):

    ax = fig.add_subplot(4, 4, i+1, xticks=[], yticks=[])

    im = Image.open(im_path)

    im = im.resize((350,480))

    plt.imshow(im)
test_jpeg = glob.glob('../input/imaterialist-fashion-2020-fgvc7/test/*')
fig = plt.figure(figsize=(25, 16))

for i,im_path in enumerate(test_jpeg[:16]):

    ax = fig.add_subplot(4, 4, i+1, xticks=[], yticks=[])

    im = Image.open(im_path)

    im = im.resize((350,480))

    plt.imshow(im)