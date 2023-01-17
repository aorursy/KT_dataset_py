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
import os

import pandas as pd

import numpy as np

import PIL.Image as Image, PIL.ImageDraw as ImageDraw, PIL.ImageFont as ImageFont
import plotly.graph_objects as go

import matplotlib.pyplot as plt
HEIGHT=137

WIDTH=236
def load_as_npa(file):

    df=pd.read_parquet(file)

    return df.iloc[:,0],df.iloc[:,1:].values.reshape(-1,HEIGHT, WIDTH)
def image_from_char(char):

    image = Image.new('RGB', (WIDTH, HEIGHT))

    draw = ImageDraw.Draw(image)

    myfont = ImageFont.truetype('/kaggle/input/bengali-fonts/hind_siliguri_normal_500.ttf', 120)

    w, h = draw.textsize(char, font=myfont)

    draw.text(((WIDTH - w) / 2,(HEIGHT - h) / 2), char, font=myfont)



    return image
image_ids0, images0 = load_as_npa('/kaggle/input/bengaliai-cv19/train_image_data_0.parquet')
f, ax = plt.subplots(5,5,figsize=(16,8))

ax = ax.flatten()



for i in range(25):

    ax[i].imshow(images0[i], cmap='Greys')
train_df = pd.read_csv('/kaggle/input/bengaliai-cv19/train.csv')

train_df.head()
image_ids0
train_df.shape
class_map_df = pd.read_csv('/kaggle/input/bengaliai-cv19/class_map.csv')

class_map_df.head()
print("Number of unique grapheme_root: {}".format(train_df['grapheme_root'].nunique()))
fig = go.Figure(data=[go.Histogram(x=train_df['grapheme_root'])])

fig.update_layout(title_text='`grapheme_root`values')

fig.show()
x = train_df['grapheme_root'].value_counts().sort_values()[-20:].index
y = train_df['grapheme_root'].value_counts().sort_values()[-20:].values
fig=go.Figure(data=[go.Bar(x=x,y=y)])

fig.update_layout(title_text='Most Common grapheme_root values.Top 20')

fig.show()
common_gr=class_map_df[(class_map_df['component_type']=='grapheme_root')&(class_map_df['label'].isin(x))]['component']
f, ax = plt.subplots(4,5,figsize=(16,8))

ax = ax.flatten()

for i in range(20):

    ax[i].imshow(image_from_char(common_gr.values[i]),cmap='Greys')
x = train_df['grapheme_root'].value_counts().sort_values()[:20].index

y = train_df['grapheme_root'].value_counts().sort_values()[:20].values

fig = go.Figure(data=[go.Bar(x=x,y=y)])

fig.update_layout(title_text='Least common grapheme_root values')

fig.show()
notcommon_gr=class_map_df[(class_map_df['component_type']=='grapheme_root')&(class_map_df['label'].isin(x))]['component']
f, ax = plt.subplots(4,5,figsize=(16,8))

ax = ax.flatten()



for i in range(20):

    ax[i].imshow(image_from_char(notcommon_gr.values[i]),cmap='Greys')
train_df['vowel_diacritic'].nunique()
x=train_df['vowel_diacritic'].value_counts().sort_values().index

y=train_df['vowel_diacritic'].value_counts().sort_values().values

fig = go.Figure(data=[go.Bar(x=x,y=y)])

fig.update_layout(title_text='Vowel_diacritic values')

fig.show()
#train Data의 component

train_df
#각 component의 label과 생김새

class_map_df
vowels = class_map_df[(class_map_df['component_type']=='vowel_diacritic')&(class_map_df['label'].isin(x))]['component']
f,ax = plt.subplots(3,5,figsize=(16,8))

ax = ax.flatten()



for i in range(15):

    if i < len(vowels):

        ax[i].imshow(image_from_char(vowels.values[i]),cmap='Greys')
vowels.values
train_df['consonant_diacritic'].nunique()
x = train_df['consonant_diacritic'].value_counts().sort_values().index

y = train_df['consonant_diacritic'].value_counts().sort_values().values

fig = go.Figure(data=[go.Bar(x=x,y=y)])

fig.update_layout(title_text='consonant_diacritic values')

fig.show()
consonants=class_map_df[(class_map_df['component_type']=='consonant_diacritic')&(class_map_df['label'].isin(x))]['component']
len(consonants)
f,ax = plt.subplots(1,7,figsize=(16,8))

ax = ax.flatten()



for i in range(7):

    ax[i].imshow(image_from_char(consonants.values[i]),cmap='Greys')
common_gr.values[9]
plt.imshow(image_from_char(common_gr.values[9]),cmap='Greys')
train_df = train_df[0:50000]



gr_root_component=class_map_df[(class_map_df['component_type']=='grapheme_root')&(class_map_df['label']==72)]['component']
class_map_df.head()
type(class_map_df)
type(class_map_df['component'])
gr_root_component
samples=train_df[train_df['grapheme_root']==72].sample(n=25)
samples.reset_index(drop=True, inplace=True)
f,ax = plt.subplots(5,5,figsize=(16,8))

ax=ax.flatten()

k=0

for i, row in samples.iterrows():

    ax[k].imshow(image_from_char(row['grapheme']),cmap='Greys')

    k=k+1
f,ax = plt.subplots(5,5,figsize=(16,8))

ax=ax.flatten()

k=0

for i, row in samples.iterrows():

    ax[k].imshow(images0[i], cmap='Greys')

    k=k+1
for i, row in samples.iterrows():

    print(i)

    print(row)
samples=train_df[

    (train_df['grapheme_root']==72)&

    (train_df['vowel_diacritic']==0)&

    (train_df['consonant_diacritic']==0)

].sample(n=25)



f, ax = plt.subplots(5,5,figsize=(16,8))

ax = ax.flatten()

k=0

for i, row in samples.iterrows():

    ax[k].imshow(images0[i],cmap='Greys')

    k=k+1