import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sb

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df_train = pd.read_csv('/kaggle/input/bengaliai-cv19/train.csv')

print('Train Data Shape: ', df_train.shape)

df_train.head()
df_test = pd.read_csv('/kaggle/input/bengaliai-cv19/test.csv')

print('Test Data Shape: ', df_test.shape)

df_test.head()
class_map = pd.read_csv('/kaggle/input/bengaliai-cv19/class_map.csv')

print('Test Data Shape: ', class_map.shape)

class_map.head()
HEIGHT = 137

WIDTH = 236



def load_npa(file):

    df = pd.read_parquet(file)

    return df.iloc[:, 1:].values.reshape(-1, HEIGHT, WIDTH)

## loading one of the parquest file for analysis

dummy_images = load_npa('/kaggle/input/bengaliai-cv19/train_image_data_0.parquet')

print("Shape of loaded files: ", dummy_images.shape)

print("Number of images in loaded files: ", dummy_images.shape[0])

print("Shape of first loaded image: ", dummy_images[0].shape)

print("\n\nFirst image looks like:\n\n", dummy_images[0])
## View the pixel values as image

plt.imshow(dummy_images[0], cmap='Greys')
f, ax = plt.subplots(5, 5, figsize=(16, 8))

for i in range(5):

    for j in range(5):

        ax[i][j].imshow(dummy_images[i*5+j], cmap='Greys')
df_train.head()
print("Unique Grapheme-Root in train data: ", df_train.grapheme_root.nunique())

print("Unique Vowel-Diacritic in train data: ", df_train.vowel_diacritic.nunique())

print("Unique Consonant-Diacritic in train data: ", df_train.consonant_diacritic.nunique())

print("Unique Grapheme (Combination of three) in train data: ", df_train.grapheme.nunique())
### Majority of Images per Grapheme count is below 180. Only 1 grapheme has 283 images in it.

images_per_grapheme = df_train.groupby('grapheme')[['image_id']].count().reset_index().reset_index()

sb.catplot(x='index', y='image_id', data=images_per_grapheme)
images_per_grapheme_root = df_train.groupby('grapheme_root')[['image_id']].count().reset_index().reset_index()

sb.catplot(x='index', y='image_id', data=images_per_grapheme_root)

images_per_grapheme_diacritic = df_train.groupby('vowel_diacritic')[['image_id']].count().reset_index()

sb.catplot(x='vowel_diacritic', y='image_id', data=images_per_grapheme_diacritic)
images_per_grapheme_diacritic = df_train.groupby('consonant_diacritic')[['image_id']].count().reset_index()

sb.catplot(x='consonant_diacritic', y='image_id', data=images_per_grapheme_diacritic)