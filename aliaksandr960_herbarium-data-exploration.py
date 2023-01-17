import os.path
import json
import codecs
from collections import Counter
import random
import math

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch.utils.data as D

from sklearn.model_selection import train_test_split
TRAIN_PATH = "../input/herbarium-2020-fgvc7/nybg2020/train/"
TRAIN_META_PATH = "../input/herbarium-2020-fgvc7/nybg2020/train/metadata.json"

TEST_PATH = "../input/herbarium-2020-fgvc7/nybg2020/test/"
TEST_META_PATH = "../input/herbarium-2020-fgvc7/nybg2020/test/metadata.json"

SUBMISSION_PATH = '../input/herbarium-2020-fgvc7/sample_submission.csv'


with codecs.open(TRAIN_META_PATH, 'r', encoding='utf-8', errors='ignore') as f:
    train_meta = json.load(f)
    
with codecs.open(TEST_META_PATH, 'r', encoding='utf-8', errors='ignore') as f:
    test_meta = json.load(f)
print('Train keys: ', train_meta.keys())
print('Test keys: ', test_meta.keys())
train_df = pd.DataFrame(train_meta['annotations'])
display(train_df)
train_cat = pd.DataFrame(train_meta['categories'])
train_cat.columns = ['family', 'genus', 'category_id', 'category_name']
display(train_cat)
train_img = pd.DataFrame(train_meta['images'])
train_img.columns = ['file_name', 'height', 'image_id', 'license', 'width']
display(train_img)
train_reg = pd.DataFrame(train_meta['regions'])
train_reg.columns = ['region_id', 'region_name', ]
display(train_reg)
train_df = train_df.merge(train_cat, on='category_id', how='outer')
train_df = train_df.merge(train_img, on='image_id', how='outer')
train_df = train_df.merge(train_reg, on='region_id', how='outer')
display(train_df)
print(train_meta['info'])
test_df = pd.DataFrame(test_meta['images'])
display(test_df)
sample_sub = pd.read_csv(SUBMISSION_PATH)
display(sample_sub)
heights = [int(w) for w in train_df['height'] if isinstance(w, float) and not math.isnan(w)]
h, b = np.histogram(heights, bins=len(set(widths)))
fig = plt.figure(figsize = (25, 5))
ax = fig.gca()
plt.plot(b[1:], h)
plt.grid()
plt.show()
widths = [int(w) for w in train_df['width'] if isinstance(w, float) and not math.isnan(w)]
h, b = np.histogram(widths, bins=len(set(widths)))
fig = plt.figure(figsize = (25, 5))
ax = fig.gca()
plt.plot(b[1:], h)
plt.grid()
plt.show()
h, b = np.histogram(train_df['category_id'], bins=len(np.unique(train_df['category_id'])))
h.sort()
fig = plt.figure(figsize = (25, 5))
ax = fig.gca()
plt.plot(h[::-1])
plt.grid()
plt.show()
GENUS_INDEX = 5

counts = list(Counter(train_df.iloc[:, GENUS_INDEX]).values())
counts.sort()
counts.reverse()

fig = plt.figure(figsize = (25, 5))
ax = fig.gca()
plt.plot(counts)
plt.grid()
plt.show()
FAMILY_INDEX = 4

counts = list(Counter(train_df.iloc[:, FAMILY_INDEX]).values())
counts.sort()
counts.reverse()

fig = plt.figure(figsize = (25, 5))
ax = fig.gca()
plt.plot(counts)
plt.grid()
plt.show()
class HerbariumDataset(D.Dataset):
    def __init__(self, data, path):
        self.data = data
        self.path = path

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        fname = self.data['file_name'].values[i]
        fpath = os.path.join(self.path, fname)
        image = cv2.imread(fpath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        label = self.data['category_id'].values[i]
        
        return image, label
train_data, test_data = train_test_split(train_df)

train_dataset = HerbariumDataset(train_data, TRAIN_PATH)
test_dataset = HerbariumDataset(test_data, TRAIN_PATH)  # There should be train path, it is correct
img, label = train_dataset[random.randint(0, len(train_dataset))]
print(label)
plt.imshow(img)
img, label = test_dataset[random.randint(0, len(test_dataset))]
print(label)
plt.imshow(img)