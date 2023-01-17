import numpy as np 

import pandas as pd 

import os

import glob

import json

from matplotlib import pyplot as plt

import seaborn as sns

from collections import Counter

import gc

from PIL import Image



%matplotlib inline
!ls /kaggle/input/imet-2020-fgvc7/
train_df = pd.read_csv("/kaggle/input/imet-2020-fgvc7/train.csv")

test_df = pd.read_csv("/kaggle/input/imet-2020-fgvc7/sample_submission.csv")

labels_df = pd.read_csv("/kaggle/input/imet-2020-fgvc7/labels.csv")
print("Train", train_df.shape)

train_df.sample(10).head()
print("Test", test_df.shape)

test_df.sample(10).head()
labels_df.sample(10).head(10)
labels_df["attribute_type"] = labels_df.attribute_name.apply(lambda x: x.split("::")[0])

print(labels_df["attribute_type"].value_counts())

sns.countplot(labels_df.attribute_type)
labels_df.attribute_id.nunique()
1920 + 768 + 681 + 100 + 5 # number of attributes = N_CLASSES
labels_df.attribute_type.unique()
labels_df[labels_df.attribute_type == "tags"]
# https://www.kaggle.com/ttahara/eda-compare-number-of-culture-and-tag-attributes

train_attr_ohot = np.zeros((len(train_df), len(labels_df)), dtype=int)



for idx, attr_arr in enumerate(train_df.attribute_ids.str.split(" ").apply(lambda l: list(map(int, l))).values):

    train_attr_ohot[idx, attr_arr] = 1

    

names_arr = labels_df.attribute_name.values

train_df["attribute_names"] = [", ".join(names_arr[arr == 1]) for arr in train_attr_ohot]



train_df["attr_num"] = train_attr_ohot.sum(axis=1)

train_df["culture_attr_num"] = train_attr_ohot[:, :398].sum(axis=1)

train_df["tag_attr_num"] = train_attr_ohot[:, 398:].sum(axis=1)
# https://www.kaggle.com/ttahara/eda-compare-number-of-culture-and-tag-attributes

fig = plt.figure(figsize=(5 * 5, 5 * 6))

fig.subplots_adjust(wspace=0.5, hspace=0.5)

for i, (art_id, attr_names) in enumerate(train_df.sort_values(by="culture_attr_num", ascending=False)[["id", "attribute_names"]].values[:15]):

    ax = fig.add_subplot(5, 3, i // 3 * 3 + i % 3 + 1)

    im = Image.open("/kaggle/input/imet-2020-fgvc7/train/{}.png".format(art_id))

    ax.imshow(im)

    im.close()

    attr_split = attr_names.split(", ")

    attr_culture = list(map(lambda x: x.split("::")[-1], filter(lambda x: x[:7] == "culture", attr_split)))

    attr_tag = list(map(lambda x: x.split("::")[-1], filter(lambda x: x[:3] == "tag", attr_split)))

    ax.set_title("art id: {}\nculture: {}\ntag: {}".format(art_id, attr_culture, attr_tag))
# https://www.kaggle.com/ttahara/eda-compare-number-of-culture-and-tag-attributes

fig = plt.figure(figsize=(5 * 6, 5 * 5))

fig.subplots_adjust(wspace=0.6, hspace=0.6)

for i, (art_id, attr_names) in enumerate(train_df.sort_values(by="tag_attr_num", ascending=False)[["id", "attribute_names"]].values[:12]):

    ax = fig.add_subplot(4, 3, i // 3 * 3 + i % 3 + 1)

    im = Image.open("/kaggle/input/imet-2020-fgvc7/train/{}.png".format(art_id))

    ax.imshow(im)

    im.close()

    attr_split = attr_names.split(", ")

    attr_culture = list(map(lambda x: x.split("::")[-1], filter(lambda x: x[:6] == "culture", attr_split)))

    attr_tag = list(map(lambda x: x.split("::")[-1], filter(lambda x: x[:3] == "tag", attr_split)))

    ax.set_title("art id: {}\nculture: {}\ntag: {}".format(art_id, attr_culture, attr_tag))
# https://www.kaggle.com/ttahara/eda-compare-number-of-culture-and-tag-attributes

fig = plt.figure(figsize=(5 * 8, 5 * 7))

fig.subplots_adjust(wspace=0.6, hspace=0.6)

for i, (art_id, attr_names) in enumerate(train_df[train_df.tag_attr_num == 1][["id", "attribute_names"]].values[:49]):

    ax = fig.add_subplot(7, 7, i // 7 * 7 + i % 7 + 1)

    im = Image.open("/kaggle/input/imet-2020-fgvc7/train/{}.png".format(art_id))

    ax.imshow(im)

    im.close()

    attr_split = attr_names.split(", ")

    attr_culture = list(map(lambda x: x.split("::")[-1], filter(lambda x: x[:7] == "culture", attr_split)))

    attr_tag = list(map(lambda x: x.split("::")[-1], filter(lambda x: x[:3] == "tag", attr_split)))

    ax.set_title("art id: {}\nculture: {}\ntag: {}".format(art_id, attr_culture, attr_tag))
fig = plt.figure(figsize=(5 * 8, 5 * 7))

fig.subplots_adjust(wspace=0.6, hspace=0.6)

for i, (art_id, attr_names) in enumerate(train_df[train_df.tag_attr_num == 2][["id", "attribute_names"]].values[:49]):

    ax = fig.add_subplot(7, 7, i // 7 * 7 + i % 7 + 1)

    im = Image.open("/kaggle/input/imet-2020-fgvc7/train/{}.png".format(art_id))

    ax.imshow(im)

    im.close()

    attr_split = attr_names.split(", ")

    attr_culture = list(map(lambda x: x.split("::")[-1], filter(lambda x: x[:7] == "culture", attr_split)))

    attr_tag = list(map(lambda x: x.split("::")[-1], filter(lambda x: x[:3] == "tag", attr_split)))

    ax.set_title("art id: {}\nculture: {}\ntag: {}".format(art_id, attr_culture, attr_tag))
fig = plt.figure(figsize=(5 * 8, 5 * 7))

fig.subplots_adjust(wspace=0.6, hspace=0.6)

for i, (art_id, attr_names) in enumerate(train_df[train_df.tag_attr_num == 3][["id", "attribute_names"]].values[:49]):

    ax = fig.add_subplot(7, 7, i // 7 * 7 + i % 7 + 1)

    im = Image.open("/kaggle/input/imet-2020-fgvc7/train/{}.png".format(art_id))

    ax.imshow(im)

    im.close()

    attr_split = attr_names.split(", ")

    attr_culture = list(map(lambda x: x.split("::")[-1], filter(lambda x: x[:7] == "culture", attr_split)))

    attr_tag = list(map(lambda x: x.split("::")[-1], filter(lambda x: x[:3] == "tag", attr_split)))

    ax.set_title("art id: {}\nculture: {}\ntag: {}".format(art_id, attr_culture, attr_tag))
train20 = pd.read_csv("/kaggle/input/imet-2020-fgvc7/train.csv")

test20 = pd.read_csv("/kaggle/input/imet-2020-fgvc7/sample_submission.csv")

labels20 = pd.read_csv("/kaggle/input/imet-2020-fgvc7/labels.csv")
train19 = pd.read_csv("/kaggle/input/imet-2019-fgvc6/train.csv")

test19 = pd.read_csv("/kaggle/input/imet-2019-fgvc6/sample_submission.csv")

labels19 = pd.read_csv("/kaggle/input/imet-2019-fgvc6/labels.csv")
train20.shape, train19.shape
np.intersect1d(train20.id.values , train19.id.values)
train19.head()
labels19.head()
labels19["attribute_type"] = labels19.attribute_name.apply(lambda x: x.split("::")[0])

print(labels19["attribute_type"].value_counts())

sns.countplot(labels19.attribute_type)
print (labels20.attribute_name.nunique())

print (labels19.attribute_name.nunique())

print (np.intersect1d(labels20.attribute_name.values , labels19.attribute_name.values).shape)