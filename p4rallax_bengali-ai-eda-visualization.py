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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import PIL.Image as Image, PIL.ImageDraw as ImageDraw, PIL.ImageFont as ImageFont

import plotly.graph_objects as go

from scipy.stats import norm
HEIGHT = 137

WIDTH = 236
def load_as_npa(file):

    df = pd.read_parquet(file)

    return df.iloc[:, 1:].values.reshape(-1 , HEIGHT , WIDTH)
def image_from_char(char):

    image = Image.new('RGB', (WIDTH, HEIGHT))

    draw = ImageDraw.Draw(image)

    myfont = ImageFont.truetype('/kaggle/input/bengalifont/HindSiliguri.ttf', 120)

    w, h = draw.textsize(char, font=myfont)

    draw.text(((WIDTH - w) / 2,(HEIGHT - h) / 2), char, font=myfont)



    return image
%%time

images0 = load_as_npa('/kaggle/input/bengaliai-cv19/train_image_data_0.parquet')
fig , ax = plt.subplots(5,5,figsize = (15,10))

ax = ax.flatten()



for i in range(25):

    ax[i].imshow(images0[i], cmap='Greys')

df_train = pd.read_csv('/kaggle/input/bengaliai-cv19/train.csv')

df_train.head()
df_train.shape
df_classmap = pd.read_csv('/kaggle/input/bengaliai-cv19/class_map.csv')

df_classmap.head()
df_classmap.shape
print("Unique grapheme roots:" ,df_train['grapheme_root'].nunique())
sns.set(rc={'figure.figsize':(10,10)})
sns.distplot(df_train['grapheme_root'] ,fit=norm , kde=False)
sns.distplot(df_train['grapheme_root'] ,kde=False)
sns.kdeplot(df_train['grapheme_root'] , shade=True)
print("Unique vowel diacritcs : ",df_train['vowel_diacritic'].nunique())
sns.distplot(df_train['vowel_diacritic'] , kde=False )
sns.kdeplot(df_train['vowel_diacritic'] , shade=True)
x = df_train['vowel_diacritic'].value_counts().sort_values().index

vowels = df_classmap[(df_classmap['component_type'] == 'vowel_diacritic') & (df_classmap['label'].isin(x))]['component']
fig, ax = plt.subplots(3, 5, figsize=(15, 10))

ax = ax.flatten()



for i in range(15):

    if i < len(vowels):

        ax[i].imshow(image_from_char(vowels.values[i]), cmap='Greys' )

        ax[i].grid(None)

        
print("Unique consonant diacritcs : ",df_train['consonant_diacritic'].nunique())
sns.distplot(df_train['consonant_diacritic'] , kde=False )
sns.kdeplot(df_train['consonant_diacritic'] , shade=True )
y = df_train['consonant_diacritic'].value_counts().sort_values().index

consonants = df_classmap[(df_classmap['component_type'] == 'consonant_diacritic') & (df_classmap['label'].isin(y))]['component']
fig, ax = plt.subplots(2, 5, figsize=(16, 10))

ax = ax.flatten()





for i in range(15):

    if i < len(consonants):

        ax[i].imshow(image_from_char(consonants.values[i]), cmap='Greys' )

        ax[i].grid(None)
df_submission = pd.read_csv('/kaggle/input/bengaliai-cv19/sample_submission.csv')

df_submission['target'] = 0

df_submission.to_csv("submission.csv", index=False)