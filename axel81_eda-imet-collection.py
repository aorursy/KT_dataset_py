import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np 

import pandas as pd 

import gc

import os

import PIL



from scipy import stats

from multiprocessing import Pool

from PIL import ImageOps, ImageFilter

from tqdm import tqdm

from wordcloud import WordCloud

from textwrap import wrap





tqdm.pandas()
train_path = '../input/train/'

train_df = pd.read_csv('../input/train.csv')

label_df = pd.read_csv('../input/labels.csv')



print('Files loaded!')
train_df['num_labels'] = train_df['attribute_ids'].apply(lambda x: len(x.split()))
train_df.head()
label_df.head()
train_df.shape, label_df.shape
category_map = {}



def fetch_categories(x):

    categories = x.split()

    for category in categories:

        category = int(category)

        if category not in category_map.keys():

                category_map[category] = 0

        category_map[category] += 1 

col = train_df['attribute_ids'].apply(lambda x: fetch_categories(x))
category_df = pd.DataFrame.from_dict(category_map, orient='index', columns=['count']).reset_index().rename(

    columns={'index': 'attribute_id'})

category_df.head()
label_df = label_df.merge(category_df, how='left', on='attribute_id')

label_df = label_df.sort_values(by='count', ascending=False)

label_df.head()
top_20_samples = label_df[:20].copy().reset_index()

plt.figure(figsize=(10,5))

sns.barplot(top_20_samples['count'], top_20_samples.index, orient='h')

plt.title('Number of samples for category (top 20)')

plt.xlabel('Number of samples')

plt.ylabel('Categories')

plt.show()
label_df['is_culture'] = label_df['attribute_name'].apply(lambda x: 1 if 'culture' in x else 0)

attribute_count = label_df['is_culture'].value_counts()



sns.barplot(['Tag', 'Culture'], attribute_count.values, alpha=0.8)

plt.title('Culture/Tag')

plt.xlabel('attribute type')

plt.ylabel('Frequency')
culture_df = label_df[label_df['is_culture'] == 1].copy()

tag_df = label_df[label_df['is_culture'] != 1].copy()
plt.figure(figsize=(20,15))



plt.subplot(1,2,1)

ax1 = sns.barplot(x=culture_df[:20]['count'],

                  y=culture_df[:20]['attribute_name'], orient="h")

plt.title('Label Counts by Culture (Top 20)',fontsize=15)

plt.xlim((0, culture_df['count'].max()*1.15))

plt.yticks(fontsize=15)



for p in ax1.patches:

    ax1.annotate(f'{int(p.get_width())}\n{p.get_width() * 100 / train_df.shape[0]:.2f}%',

                (p.get_width(), p.get_y() + p.get_height() / 2.), 

                ha='left', 

                va='center', 

                fontsize=12, 

                color='black',

                xytext=(7,0), 

                textcoords='offset points')



plt.subplot(1,2,2)    

ax2 = sns.barplot(x=tag_df[:20]['count'],

                  y=tag_df[:20]['attribute_name'], orient="h")

plt.title('Label Counts by Tag (Top 20)',fontsize=15)

plt.xlim((0, tag_df['count'].max()*1.15))

plt.yticks(fontsize=15)



for p in ax2.patches:

    ax2.annotate(f'{int(p.get_width())}\n{p.get_width() * 100 / train_df.shape[0]:.2f}%',

                (p.get_width(), p.get_y() + p.get_height() / 2.), 

                ha='left', 

                va='center', 

                fontsize=12, 

                color='black',

                xytext=(7,0), 

                textcoords='offset points')



plt.tight_layout()

plt.show()

def plot_image_for_attribute(attribute_id, idx, rows=10, cols=3):

    global cnt

    plt.figure(figsize=(15,6))



    # str contains logic would fail for ids <=109 because of regex matching

    img_ids = train_df[train_df['attribute_ids'].str.contains(attribute_id)][:cols][['id', 'attribute_ids']]



    for img_id, attr_ids in zip(img_ids['id'].values, img_ids['attribute_ids'].values):

        attr_ids = attr_ids.split()

        title = ''

        for attr_id in attr_ids:

            title += label_df.loc[int(attr_id)]['attribute_name'] + ' '



        img = PIL.Image.open(f'{train_path}{img_id}.png')

        plt.subplot(rows, cols, idx)

        plt.imshow(img)

        plt.axis('off')

        plt.title(("\n".join(wrap(title, 40))))

        idx += 1



def plot_img_for_attribute_ids(attribute_ids, num_imgs=3):

    n_rows = len(attribute_ids)

    n_cols = num_imgs

    idx = 1

    for attr_id in attribute_ids:

        print(attr_id, idx)

        plot_image_for_attribute(str(attr_id), idx, n_rows, n_cols)

        idx += n_cols 

    plt.show()
attr_ids = culture_df[:2]['attribute_id'].values

plot_img_for_attribute_ids(attr_ids)
attr_ids = tag_df[:2]['attribute_id'].values

plot_img_for_attribute_ids(attr_ids)
plt.figure(figsize=(20,8))



ax = sns.countplot(train_df['num_labels'])

plt.xlabel('Number of Labels')

plt.title('Number of Labels per Image', fontsize=20)



for p in ax.patches:

    ax.annotate(f'{p.get_height() * 100 / train_df.shape[0]:.3f}%',

            (p.get_x() + p.get_width() / 2., p.get_height()), 

            ha='center', 

            va='center', 

            fontsize=11, 

            color='black',

            xytext=(0,7), 

            textcoords='offset points')
img_ids = train_df.groupby(by='num_labels').first().reset_index()

img_ids = img_ids[img_ids['num_labels']>6]

img_ids.shape
img_ids.head()
plt.figure(figsize=(20, 10))

idx = 1

for img_id, attr_ids in zip(img_ids['id'].values, img_ids['attribute_ids'].values):

        attr_ids = attr_ids.split()

        title = ''

        for attr_id in attr_ids:

            title += label_df.loc[int(attr_id)]['attribute_name'] + ' '



        img = PIL.Image.open(f'{train_path}{img_id}.png')

        plt.subplot(3, 3, idx)

        plt.imshow(img)

        plt.axis('off')

        plt.title(("\n".join(wrap(title, 40))))

        idx += 1

plt.subplots_adjust(hspace=1, wspace=0)

plt.show()
img_ids = label_df[label_df['count'] == 1]

print(img_ids.shape)

img_ids.head(15)
img_ids = label_df[label_df['count'] <= 5]

print(img_ids.shape)

img_ids.head()
sparse_data_imgs = img_ids.groupby('count')['is_culture'].value_counts().to_frame().reset_index(level='count').rename(columns={

    'is_culture': 'is_culture_count', 'count': 'num_labels'}).reset_index()



sns.barplot(x='num_labels', y='is_culture_count', data=sparse_data_imgs, hue='is_culture', alpha=0.8)

plt.title('Culture/Tag')

plt.xlabel('Num Labels')

plt.ylabel('Frequency')