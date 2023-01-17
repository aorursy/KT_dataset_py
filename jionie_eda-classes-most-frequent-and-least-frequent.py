# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



import numpy as np

import pandas as pd

import os

import copy

import sys

from PIL import Image

import time 

from tqdm.autonotebook import tqdm

import random

import gc

import cv2

import scipy

import math

import matplotlib.pyplot as plt



from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold,StratifiedKFold

from sklearn.metrics import fbeta_score



import torch

from torch.utils.data import TensorDataset, DataLoader,Dataset

import torch.nn as nn

import torch.nn.functional as F

import torchvision

import torchvision.transforms as transforms

import torch.optim as optim

from torch.optim import lr_scheduler

from torch.optim.optimizer import Optimizer

import torch.backends.cudnn as cudnn

from torch.autograd import Variable

from torch.utils.data.sampler import SubsetRandomSampler

from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR, _LRScheduler



# Any results you write to the current directory are saved as output.
import scipy.special



SEED = 42

base_dir = '../input/'

def seed_everything(seed=SEED):

    random.seed(seed)

    os.environ['PYHTONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True

seed_everything(SEED)
train_df = pd.read_csv('../input/train.csv')

labels_df = pd.read_csv('../input/labels.csv')

test_df = pd.read_csv('../input/sample_submission.csv')



tr, val = train_test_split(train_df['id'], test_size=0.15, random_state=SEED)



img_class_dict = {k:v for k, v in zip(train_df.id, train_df.attribute_ids)}



def get_label(attribute_ids):

    attribute_ids = attribute_ids.split()

    one_hot = np.zeros(1103, dtype=np.int)

    for _,ids in enumerate(attribute_ids):

        one_hot[int(ids)] = 1

    return one_hot
print(train_df.columns)

print(labels_df.columns)
classes =train_df['attribute_ids'].value_counts().to_frame().reset_index()

classes.rename(columns={'index': 'classes', 'attribute_ids':'counts'}, inplace=True)
print(classes)
#classes['classes'] = classes['classes'].apply(get_label)
classes['ratio'] = classes['counts']/train_df.shape[0]
classes.head(10)
def get_label_name(attribute_ids):

    attribute_ids = attribute_ids.split()

    attribute_name = []

    for _,ids in enumerate(attribute_ids):

        attribute_name.append(labels_df.loc[labels_df['attribute_id']==int(ids)])

    return attribute_name
#train_df['attribute_name'] = train_df['attribute_ids'].apply(get_label_name)

#too slow
train_df['count'] = train_df.groupby(['attribute_ids'])['id'].transform('count')
train_df = train_df.sort_values(by='attribute_ids')
#train_df['attribute_ids'] = train_df['attribute_ids'].apply(get_label)
train_df.head(30)
grouped_id = train_df.groupby('attribute_ids')['id']
collect_image_names = {}



for key in classes['classes']:

    name = grouped_id.get_group(key).values[0]

    count = grouped_id.get_group(key).values.shape[0]

    collect_image_names[name] = count
import operator

sorted_collect_image_names = sorted(collect_image_names.items(), key=operator.itemgetter(1))

sorted_collect_image_names.reverse()

print(len(sorted_collect_image_names))
print(sorted_collect_image_names[:10])
image_name = sorted_collect_image_names[0][0]

attribute_ids = train_df.loc[train_df['id']==image_name]['attribute_ids'].values[0]

print(attribute_ids.split())
c = 1

plt.figure(figsize=[20, 20])

for idx in range(10):

    image_name = sorted_collect_image_names[idx][0]

    img = cv2.imread("../input/train/{}.png".format(image_name))[...,[2,1,0]]

    plt.subplot(5,2,c)

    plt.imshow(img)

    

    attribute_ids = train_df.loc[train_df['id']==image_name]['attribute_ids'].values[0].split()

    attribute_name = []

    for _,ids in enumerate(attribute_ids):

        attribute_name.append(labels_df.loc[labels_df['attribute_id']==int(ids)]['attribute_name'].values[0])

    plt.title("train image {} count {}".format(attribute_name, sorted_collect_image_names[idx][1]))

    c += 1

plt.show()
c = 1

plt.figure(figsize=[20,20])



size = len(sorted_collect_image_names)



for idx in range(size-10, size):

    image_name = sorted_collect_image_names[idx][0]

    img = cv2.imread("../input/train/{}.png".format(image_name))[...,[2,1,0]]

    plt.subplot(5,2,c)

    plt.imshow(img)

    

    attribute_ids = train_df.loc[train_df['id']==image_name]['attribute_ids'].values[0].split()

    attribute_name = []

    for _,ids in enumerate(attribute_ids):

        attribute_name.append(labels_df.loc[labels_df['attribute_id']==int(ids)]['attribute_name'].values[0])

    plt.title("train image {} count {}".format(attribute_name, sorted_collect_image_names[idx][1]))

    c += 1

plt.show()
name = grouped_id.get_group(classes['classes'][0]).values[0]

count = grouped_id.get_group(classes['classes'][0]).values.shape[0]
c = 1

plt.figure(figsize=[20,20])



most_frequent_class_top_10 = {}



for i in range(10):

    name = grouped_id.get_group(classes['classes'][0]).values[i]

    count = grouped_id.get_group(classes['classes'][0]).values.shape[0]

    most_frequent_class_top_10[name] = count



size = len(most_frequent_class_top_10)



for element in most_frequent_class_top_10:

    image_name = element

    img = cv2.imread("../input/train/{}.png".format(image_name))[...,[2,1,0]]

    plt.subplot(5,2,c)

    plt.imshow(img)

    

    attribute_ids = train_df.loc[train_df['id']==image_name]['attribute_ids'].values[0].split()

    attribute_name = []

    for _,ids in enumerate(attribute_ids):

        attribute_name.append(labels_df.loc[labels_df['attribute_id']==int(ids)]['attribute_name'].values[0])

    plt.title("train image {} count {}".format(attribute_name, most_frequent_class_top_10[element]))

    c += 1

plt.show()
category_count = {}



for i in range(1103):

    category_count[i] = 0
for key in classes['classes']:

    category_name = key.split()

    count = grouped_id.get_group(key).values.shape[0]

    for element in category_name:

        category_count[int(element)] += count
sorted_category_count = sorted(category_count.items(), key=operator.itemgetter(1))

sorted_category_count.reverse()
sorted_category_count_frame = pd.DataFrame.from_dict(sorted_category_count)

sorted_category_count_frame.columns=['attribute_id', 'count']

sorted_category_count_frame['ratio'] = sorted_category_count_frame['count']/train_df.shape[0]
sorted_category_count_frame.head(30)
category_name_count = {}



for element in sorted_category_count:

    key = element[0]

    name = labels_df[labels_df['attribute_id']==key]['attribute_name'].values[0]

    category_name_count[name] = element[1]
sorted_category_name_count = sorted(category_name_count.items(), key=operator.itemgetter(1))

sorted_category_name_count.reverse()
sorted_sorted_category_name_count_frame = pd.DataFrame.from_dict(sorted_category_name_count)

sorted_sorted_category_name_count_frame.columns=['attribute_name', 'count']

sorted_sorted_category_name_count_frame['ratio'] = sorted_sorted_category_name_count_frame['count']/train_df.shape[0]
sorted_sorted_category_name_count_frame.head(30)
sorted_category_count_frame.to_csv('sorted_category_count_frame.csv', index=False)