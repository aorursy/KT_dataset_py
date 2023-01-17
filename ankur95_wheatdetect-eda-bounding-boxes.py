# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import scipy.io

import scipy.misc

import PIL

import tensorflow as tf

import seaborn as sns

import matplotlib.image as mpimg

from collections import Counter

import os
train_names = os.listdir("/kaggle/input/global-wheat-detection/train/")
# Looking at individual image

# Change the value of i to see different images

i = 100



a = os.path.join("/kaggle/input/global-wheat-detection/train/" + train_names[i])

a = mpimg.imread(a)

plt.imshow(a)
dfo = pd.read_csv("/kaggle/input/global-wheat-detection/train.csv")

dfo.head()
dfo["x_min"] = dfo["bbox"].str[1:-1].apply(lambda x : x.split(",")[0]).astype(float)

dfo["y_min"] = dfo["bbox"].str[1:-1].apply(lambda x : x.split(",")[1]).astype(float)

dfo["width"] =  dfo["bbox"].str[1:-1].apply(lambda x : x.split(",")[2]).astype('float32')

dfo["height"] =  dfo["bbox"].str[1:-1].apply(lambda x : x.split(",")[3]).astype('float32')

dfo["x_max"] = dfo["x_min"] + dfo["width"]

dfo["y_max"] = dfo["y_min"] + dfo["height"]
dfo.head()
ex_type = []

for i in train_names:

    ex_type.append(i.split(".")[1])
Counter(ex_type)
dfo["image_id"] =  dfo["image_id"] + ".jpg"
image_ids = dfo["image_id"].unique()

image_ids = list(set(image_ids))
count_imageid = Counter(dfo["image_id"])
import matplotlib.patches as patches

from PIL import Image
# Draw BOundary BOxes

imid = 120 # changes this no to see different image



im = np.array(Image.open(os.path.join("/kaggle/input/global-wheat-detection/train/" , train_names[imid])))

fig, ax = plt.subplots(1)



ax.imshow(im)

for i in range(count_imageid[train_names[imid]]):

    x_min = dfo[dfo["image_id"] == train_names[imid]]["x_min"].iloc[i]

    y_min = dfo[dfo["image_id"] == train_names[imid]]["y_min"].iloc[i]

    width = dfo[dfo["image_id"] == train_names[imid]]["width"].iloc[i]

    height = dfo[dfo["image_id"] == train_names[imid]]["height"].iloc[i]

    rect = patches.Rectangle((x_min, y_min), width, height, facecolor = "none", linewidth = 1, edgecolor = "r")

    ax.add_patch(rect)

plt.show()