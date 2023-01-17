#!pip install -U tensorflow
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import time, gc

import tensorflow as tf

from PIL import Image

print(tf.__version__)



from sklearn.model_selection import train_test_split

from matplotlib import pyplot as plt

import matplotlib

matplotlib.use('Agg')



# import the necessary keras and sklearn packages



from sklearn.preprocessing import LabelBinarizer

from sklearn.model_selection import train_test_split



import random



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_df_ = pd.read_csv('/kaggle/input/bengaliai-cv19/train.csv')

test_df_ = pd.read_csv('/kaggle/input/bengaliai-cv19/test.csv')

class_map_df = pd.read_csv('/kaggle/input/bengaliai-cv19/class_map.csv')

sample_sub_df = pd.read_csv('/kaggle/input/bengaliai-cv19/sample_submission.csv')
train_df_.head()
img_50310 = train_df_[train_df_.image_id=='Train_50310']

img_50220 = train_df_[train_df_.image_id=='Train_50220']
img_50310
img_50220
len(train_df_)
class_map_df.head()
class_map_df.component_type.value_counts()
class_map_df_root = class_map_df[class_map_df.component_type=='grapheme_root']

class_map_df_vowel = class_map_df[class_map_df.component_type=='vowel_diacritic']

class_map_df_cons = class_map_df[class_map_df.component_type=='consonant_diacritic']
class_map_df_root.head()
class_map_df_vowel.head()
class_map_df_cons.head()
train_df_groot = train_df_.groupby(['grapheme_root']).size().reset_index()

train_df_groot=train_df_groot.rename(columns={0:'count'})
class_map_df_groot = class_map_df[class_map_df.component_type=='grapheme_root']

groot_merged = pd.merge(train_df_groot,class_map_df_groot[['label','component']],left_on='grapheme_root',right_on='label',how='inner')

groot_merged.sort_values(by="count",ascending=False)[:10]
train_df_vd = train_df_.groupby(['vowel_diacritic']).size().reset_index()

train_df_vd=train_df_vd.rename(columns={0:'count'})

class_map_df_vd = class_map_df[class_map_df.component_type=='vowel_diacritic']

vd_merged = pd.merge(train_df_vd,class_map_df_vd[['label','component']],left_on='vowel_diacritic',right_on='label',how='inner')

vd_merged.sort_values(by="count",ascending=False)
train_df_cd = train_df_.groupby(['consonant_diacritic']).size().reset_index()

train_df_cd=train_df_cd.rename(columns={0:'count'})

class_map_df_cd = class_map_df[class_map_df.component_type=='consonant_diacritic']

cd_merged = pd.merge(train_df_cd,class_map_df_cd[['label','component']],left_on='consonant_diacritic',right_on='label',how='inner')

cd_merged.sort_values(by="count",ascending=False)
def read_data(nf):

    nf=int(nf)

    train_df = pd.read_feather(f'/kaggle/input/bengaliaicv19feather/train_image_data_{nf}.feather')

    return train_df



def read_test_data(nf):

    nf=int(nf)

    test_df = pd.read_feather(f'/kaggle/input/bengaliaicv19feather/test_image_data_{nf}.feather')

    return test_df
train_df=read_data(1)
train_df.head()
len(train_df.columns)
%matplotlib inline

import sys

label = train_df.iloc[100,0]

print(label)

img = train_df.iloc[100,1:]

img=img.astype('uint8')

img = np.array(img).reshape(137,236)

plt.imshow(img);
img = train_df.iloc[10,1:]

label = train_df.iloc[10,0]

img=img.astype('uint8')

print(label)

img = np.array(img).reshape(137,236)

img = Image.fromarray(img)

plt.imshow(img);
fig = plt.figure()

img_resized = img.resize((96,96))

plt.imshow(img_resized);