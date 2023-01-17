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

class_map_df = pd.read_csv('/kaggle/input/bengaliai-cv19/class_map.csv')

sample_sub_df = pd.read_csv('/kaggle/input/bengaliai-cv19/sample_submission.csv')
train.head()
len(train)
train.tail()
test.head()
test.tail()
## 3 outputs or prediction from 1 image. So total 12 images in test set.

len(test)/3
sample_sub_df.head()
class_map_df.head()
class_map_df.component_type.unique()
class_map_df.component_type.value_counts()
class_map_df.component[class_map_df.component_type=='consonant_diacritic']
class_map_df[class_map_df.component_type=='consonant_diacritic']
const_diac = class_map_df.component[class_map_df.component_type=='consonant_diacritic'].values

const_diac
for diac in const_diac:

    for i in diac:

        print(i,end=' ')

    print()
HEIGHT = 236

WIDTH = 236

import PIL.Image as Image, PIL.ImageDraw as ImageDraw, PIL.ImageFont as ImageFont

import matplotlib.pyplot as plt



def image_from_char(char):

    image = Image.new('RGB', (WIDTH, HEIGHT))

    draw = ImageDraw.Draw(image)

    myfont = ImageFont.truetype('/kaggle/input/kalpurush-fonts/kalpurush-2.ttf', 120)

    w, h = draw.textsize(char, font=myfont)

    draw.text(((WIDTH - w) / 2,(HEIGHT - h) / 3), char, font=myfont)



    return image
f, ax = plt.subplots(1, 7, figsize=(16, 8))

ax = ax.flatten()



for i,diac in enumerate(const_diac):

    ax[i].imshow(image_from_char(diac), cmap='Greys')
class_map_df.head()
train.head()
const_sample = train.sort_values(['consonant_diacritic']).groupby('consonant_diacritic').head(2).reset_index()

const_sample
for j,i in enumerate(range(0,len(const_sample),2)):

    print(j,i)
f, ax = plt.subplots(2, 7, figsize=(16, 8))

ax = ax.flatten()



for i,diac in enumerate(const_diac):

    ax[i].axis("off")

    print(diac)

    ax[i].imshow(image_from_char(diac), cmap='Greys')

for j,i in enumerate(range(0,len(const_sample),2)):

    x = const_sample.iloc[i].grapheme

    print(x)

    ax[j+7].axis("off")

#     ax[j+7].title.set_text(x)

    ax[j+7].imshow(image_from_char(x), cmap='Greys')
class_map_df.component[class_map_df.component_type=='vowel_diacritic']
len(train), len(test)
train.head()
ok = pd.read_parquet(f'/kaggle/input/bengaliai-cv19/train_image_data_0.parquet')

ok.head()
ok = pd.merge(ok, train, on='image_id')

ok.head()
len(ok), len(train)
50210*4 # 4 parquet files
only_imgs = ok.drop(columns=['image_id','grapheme_root','vowel_diacritic','consonant_diacritic','grapheme'])

only_imgs.head()
len(only_imgs.columns)
HEIGHT = 137

WIDTH = 236

HEIGHT*WIDTH
f, ax = plt.subplots(5, 5, figsize=(16, 8))

ax = ax.flatten()



for i in range(25):

    ax[i].axis("off")

    ax[i].imshow(only_imgs.iloc[i].values.reshape(HEIGHT,WIDTH), cmap='Greys')
# for i in range(4):

#     ### inner train will remove other rows

#     train_df = pd.merge(pd.read_parquet(f'/kaggle/input/bengaliai-cv19/train_image_data_{i}.parquet'), train_df, on='image_id')#.drop(['image_id'], axis=1)