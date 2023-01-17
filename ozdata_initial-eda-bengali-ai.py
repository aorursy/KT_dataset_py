# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from matplotlib.gridspec import  GridSpec

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv('../input/bengaliai-cv19/train.csv')
train_df.tail()
test_df = pd.read_csv('../input/bengaliai-cv19/test.csv')
test_df.tail()
train_image_0 = pd.read_parquet('/kaggle/input/bengaliai-cv19/train_image_data_0.parquet')
train_image_0.head()
IMAGE_ROW = 137 

IMAGE_COLUMN = 236



train_image0 = train_image_0.drop('image_id', axis =1)



def display_image(idx): 

    img = train_image0[idx:idx+1].values[0].reshape([IMAGE_ROW, IMAGE_COLUMN])

    plt.imshow(img, cmap = 'gray')

    plt.axis('off')
train_image0[1:2].values[0]
display_image(70)
def display_range_images(idx): 

    gr, vd, cd = list(train_df.iloc[idx][['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']])

    figure, ax = plt.subplots(3,3)

    figure.tight_layout(rect=[0, 0.03, 1, 0.90])

    same_char_loc = list(train_df[(train_df['grapheme_root'] == gr) & 

                        (train_df['vowel_diacritic'] == vd) & 

                        (train_df['consonant_diacritic'] == cd)].index)[:9]

    for i in range(9):

        plt.subplot(3,3,i+1)

        plt.axis('off')

        cidx = same_char_loc[i]

        img = train_image0[cidx:cidx+1].values[0].reshape([IMAGE_ROW, IMAGE_COLUMN])

        plt.imshow(img, cmap = 'gray')

    figure.suptitle('Grapheme Root Class: {} \n Vowel Diacritic Class: {} \n Consonant Diacritic Class: {}'.format(gr,vd,cd))

    plt.show()
display_range_images(10)
list(train_df.iloc[70][['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']])
"""

grapheme_root               148

vowel_diacritic               0

consonant_diacritic           5

"""

train_df[(train_df['grapheme_root'] == 115) & 

         (train_df['vowel_diacritic'] == 1) & 

         (train_df['consonant_diacritic'] == 0)].index[:9]
no_cs = train_df[(train_df['grapheme_root'] == 115) & 

         (train_df['vowel_diacritic'] == 1) & 

         (train_df['consonant_diacritic'] == 0)].index[:2]

with_cs = train_df[(train_df['grapheme_root'] == 115) & 

         (train_df['vowel_diacritic'] == 1) & 

         (train_df['consonant_diacritic'] == 2)].index[:2]

fig, ax = plt.subplots(2,2)

fig.suptitle('Grapheme Root Class: 115 \n Vowel Diacritic Class: 1')

fig.tight_layout(rect=[0, 0.03, 1, 0.85])

for i in range(4):

    plt.subplot(2,2,i+1)

    plt.axis('off')

    if i % 2 == 0:

        idx = no_cs[i // 2]

        print(idx)

        img = train_image0[idx:idx+1].values[0].reshape([IMAGE_ROW, IMAGE_COLUMN])

        if i == 0:

            plt.title('Consonant Diacritic: 0')

        plt.imshow(img, cmap = 'gray')

    else:

        idx = with_cs[i // 2]

        print(idx)

        img = train_image0[idx:idx+1].values[0].reshape([IMAGE_ROW, IMAGE_COLUMN])

        if i == 1:

            plt.title('Consonant Diacritic: 2')

        plt.imshow(img, cmap = 'gray')

plt.show()
roots = train_df.groupby(by=['grapheme_root']).count()

roots = roots.reset_index()[['grapheme_root', 'image_id']].sort_values(by=['image_id'], ascending = False)
roots['count'] = roots['image_id']

roots = roots.drop('image_id', axis = 1)
fig,ax = plt.subplots(figsize=(15,5))

sns.barplot(x = 'grapheme_root', y='count', data = roots, order=roots['grapheme_root'])

plt.xticks(rotation=90)

fig.tight_layout()

plt.title('Number of images in the Training set across Grapheme Root classes', fontsize = 18)

plt.show()
vowels = train_df.groupby(by=['vowel_diacritic']).count()

vowels = vowels.reset_index()[['vowel_diacritic', 'image_id']].sort_values(by=['image_id'], ascending = False)

vowels['count'] = vowels['image_id']

vowels = vowels.drop('image_id', axis = 1)



consonants = train_df.groupby(by=['consonant_diacritic']).count()

consonants = consonants.reset_index()[['consonant_diacritic', 'image_id']].sort_values(by=['image_id'], ascending = False)

consonants['count'] = consonants['image_id']

consonants = consonants.drop('image_id', axis = 1)

figure,ax = plt.subplots(1,2,figsize = (8,4))

plt.subplot(121)

sns.barplot(x = 'vowel_diacritic', y='count', data = vowels, order=vowels['vowel_diacritic'])

plt.title('Number of images per Vowel Diacritic', fontsize = 12)

plt.subplot(122)

sns.barplot(x = 'consonant_diacritic', y='count', data = consonants, order=consonants['consonant_diacritic'])

plt.title('Number of images per Consonant Diacritic', fontsize = 12)

plt.tight_layout()
vowels