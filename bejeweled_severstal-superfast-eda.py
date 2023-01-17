# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
SEED = 1337

np.random.seed(SEED)



import cv2

import matplotlib.pyplot as plt

import seaborn as sns



sns.set_style("whitegrid")
train_df = pd.read_csv("../input/train.csv")

sub = pd.read_csv("../input/sample_submission.csv")



display(train_df.head())

display(sub.head())
train_df.shape, sub.shape
train_df[train_df["EncodedPixels"].isna()].shape
c = 1

plt.figure(figsize=[16,16])

for img_name in os.listdir("../input/train_images/")[:16]:

    img = cv2.imread("../input/train_images/{}".format(img_name))[...,[2,1,0]]

    plt.subplot(4,4,c)

    plt.imshow(img)

    plt.title("train img{}. shape = {}".format(c, img.shape))

    c += 1

plt.show();
c = 1

plt.figure(figsize=[16,16])

for img_name in os.listdir("../input/test_images/")[:16]:

    img = cv2.imread("../input/test_images/{}".format(img_name))[...,[2,1,0]]

    plt.subplot(4,4,c)

    plt.imshow(img)

    plt.title("test img{}. shape = {}".format(c, img.shape))

    c += 1

plt.show();
for img_name in os.listdir("../input/train_images/"):

    img = cv2.imread("../input/train_images/{}".format(img_name))[...,[2,1,0]]

    if img.shape != (256,1600,3):

        print(img.shape)
for img_name in os.listdir("../input/test_images/"):

    img = cv2.imread("../input/test_images/{}".format(img_name))[...,[2,1,0]]

    if img.shape != (256,1600,3):

        print(img.shape)