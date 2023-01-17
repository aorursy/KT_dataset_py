# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('/kaggle/input/bengaliai-cv19/train.csv')

test = pd.read_csv('/kaggle/input/bengaliai-cv19/test.csv')

sample = pd.read_csv('/kaggle/input/bengaliai-cv19/sample_submission.csv')

class_map = pd.read_csv('/kaggle/input/bengaliai-cv19/class_map.csv')

print('Size of train data', train.shape)

print('Size of test data', test.shape)

print('Size of sample submission', sample.shape)

print('Size of Class Map: ', class_map.shape)
train.head()
train.columns
train.describe()
test.head()
test.columns
test.describe()
sample.head()
class_map.head()
HEIGHT = 137

WIDTH = 236



def load_images(file):

    df = pd.read_parquet(file)

    return df.iloc[:, 1:].values.reshape(-1, HEIGHT, WIDTH)
## loading one of the parquest file for analysis

dummy_images = load_images('/kaggle/input/bengaliai-cv19/train_image_data_0.parquet')

print("Shape of loaded files: ", dummy_images.shape)

print("Number of images in loaded files: ", dummy_images.shape[0])

print("Shape of first loaded image: ", dummy_images[0].shape)

print("\n\nFirst image looks like:\n\n", dummy_images[0])
import seaborn as sb

import matplotlib.pyplot as plt



## View the pixel values as image

plt.imshow(dummy_images[10], cmap='Greys')
f, ax = plt.subplots(6, 6, figsize=(16, 10))



for i in range(6):

    for j in range(6):

        ax[i][j].imshow(dummy_images[i*6+j], cmap='Greys')

train.isnull().sum()
test.isnull().sum()
class_map.isnull().sum()
import seaborn as sns





sns.catplot(x='vowel_diacritic',data=train,kind="count", height=8.27, aspect=11.7/8.27)
sns.catplot(x='consonant_diacritic',data=train,kind="count", height=8.27, aspect=11.7/8.27)

sns.catplot(x='grapheme_root',data=train,kind="count", height=8.27, aspect=30/8.27)

print("Unique Grapheme-Root in train data: ", train.grapheme_root.nunique())

print("Unique Vowel-Diacritic in train data: ", train.vowel_diacritic.nunique())

print("Unique Consonant-Diacritic in train data: ", train.consonant_diacritic.nunique())

print("Unique Grapheme (Combination of three) in train data: ", train.grapheme.nunique())