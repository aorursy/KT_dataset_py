# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



import re

import json

import textwrap as text



import warnings

warnings.filterwarnings('ignore')



df = pd.read_csv('/kaggle/input/us-videos/USvideos.csv',

                 parse_dates=['trending_date'])



with open('/kaggle/input/categories/category_ids.json') as f:

    cat_ids = json.load(f)



# Just want { id: title }

cat_ids = { int(item['id']): item['snippet']['title'] for item in cat_ids['items'] }
most_trending = df['title'].value_counts()[:5]



print(most_trending)
df = df.drop_duplicates(subset='title', keep='last')
most = {

    'viewed': 'views',

    'commented': 'comment_count',

    'liked': 'likes',

    'disliked': 'dislikes'

}



for k, v in most.items():

    largest = df.nlargest(1, v)

    

    print(text.dedent(f"""

        Most {k.title()}: {largest['title'].values[0]}

        Channel: {largest['channel_title'].values[0]}

        Total {k.title()}: {largest[v].values[0] :,}

    """))
from collections import defaultdict

import heapq



tag_counts = defaultdict(int)



# Process a tag list (as string) and return list of tags

def process_tlist(l):

    if l == '[none]':

        return None

    

    # remove quotes and split on pipes |

    l = l.replace('"','').split('|')

    return l



for t_list in df['tags']:

    tlist = process_tlist(t_list)

    

    if tlist is None:

        continue

    

    for tag in tlist:

        tag_counts[tag] += 1



top_tags = heapq.nlargest(10, tag_counts, key=tag_counts.get)

print(f'Top Tags: {top_tags}')
import seaborn as sns 



# Map ids to category names

df['category'] = df['category_id'].map(cat_ids)



# Top categories by view count

top_categories = df['category'].value_counts()[:5]



filtered_df = df[df['category'].isin(top_categories.index)]



ax = sns.catplot(x='category',

                y='views',

                hue='category', 

                data=filtered_df,

                height=7,

                aspect=1.5)

ax.set(title='Views by Top Categories')

plt.show()

ax.savefig('/kaggle/working/ViewsByTopCategories.png')
import seaborn as sns 



# Top channels

top_channels = df['channel_title'].value_counts()[:5]



filtered_df = df[df['channel_title'].isin(top_channels.index)]



# Views by top channels

ax = sns.catplot(x='channel_title', 

                y='views', 

                hue='channel_title', 

                data=filtered_df,

                height=7,

                aspect=1.5)

ax.set(title='Views by Top Channels')



print(top_channels)

plt.show()



ax.savefig('/kaggle/working/ViewsByTopChannels.png')
# import os, shutil, requests, time



# if not os.path.exists('/kaggle/working/images/'):

#     os.mkdir('/kaggle/working/images/')



# # Mildly edited from https://www.kaggle.com/abinesh100/easy-download-images-in-25-lines-py3

# # Downloads images into input/images/image.jpg

# def fetch_image(path):

#     url=path

#     response=requests.get(url, stream=True)

#     with open('/kaggle/working/images/image.jpg', 'wb') as out_file:

#         shutil.copyfileobj(response.raw, out_file)

#     del response

    

# links = df['thumbnail_link']



# print('fetching images...')

# i = 0

# for link in links:

#     # Prevent unneccessary redownload of images

# #     if os.path.exists(f'/kaggle/working/images/{i}.jpg'):

# #         i += 1

# #         continue

        

#     fetch_image(link)

#     os.rename('/kaggle/working/images/image.jpg', f'/kaggle/working/images/{i}.jpg')

    

#     i += 1

    

# print('images fetched!')
import cv2

import os



# cv2 cascade classifier path

# Using most relaxed default

classifier_path = '../input/haar-cascades-for-face-detection/haarcascade_frontalface_default.xml'

classifier = cv2.CascadeClassifier(classifier_path)



# Generator of image paths

# Default length is all the images in /images/

# Bad practice :( but simple to code

def image_paths(length=False):

    length = len(df) if not length else length

    

    for i in range(length):

        yield f'../input/youtube-thumbnails/{i}.jpg'



# Tweaking values with known facial images

# known = [0, 3, 4, 5]

# for n in known:

#     image = cv2.imread(f'/kaggle/working/images/{n}.jpg')

#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)



#     faces = classifier.detectMultiScale(

#         gray,

#         minNeighbors=2,

#         minSize=(5, 5),

#     )



#     print(f'Image {n}, Detected Faces: {len(faces)}')

has_face = []



print('classifying images')

for image_path in image_paths():

    image = cv2.imread(image_path)

    

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                        

    faces = classifier.detectMultiScale(

        gray,

        minNeighbors=5,

        minSize=(5, 5),

    )

    

    has_face.append(len(faces) > 0)

print('done classifying')



# len(has_face) == len(df) # sanity check
df['has_face'] = pd.Series(has_face)



# Top categories by view count

top_categories = df['category'].value_counts()[:6]



filtered_df = df[df['category'].isin(top_categories.index)]



face_mask = filtered_df['has_face'] == True



for cat in top_categories.index:

    cat_mask = filtered_df['category'] == cat

    yes = len(filtered_df[face_mask][cat_mask])

    no = len(filtered_df.drop(filtered_df[face_mask].index)[cat_mask])

    print(text.dedent(f'''

        Category: {cat}

        Has (at least 1) face: {yes}

        No face: {no}

        Proportion: {yes/no :.2f}

    '''))



sns.set(font_scale=1.2)

g = sns.catplot(

    x = 'has_face',

    y = 'views',

    col = 'category',

    col_wrap = 3,

    data = filtered_df

)

g.savefig('/kaggle/working/ViewsByCategoryAndFaces.png')
# Category: Howto & Style

g = sns.catplot(

    x = 'category',

    y = 'views',

    hue = 'has_face',

    data = filtered_df[filtered_df['category'] == 'Howto & Style']

)
# Category: People & Blogs

g = sns.catplot(

    x = 'category',

    y = 'views',

    hue = 'has_face',

    data = filtered_df[filtered_df['category'] == 'People & Blogs']

)
import re

from functools import reduce



# Return proportion of capital words (converted to characters) to total characters

# Ignores singular I's to remove some false positives

def proportion(s):

    # length of all capital words in a string

    # splits string on non-word characters

    all_caps = [ w for w in re.split(r"\W", s.replace('"', '')) if w.isupper() and w != 'I' ]

    total_caps = reduce(lambda x, y: x + len(y), all_caps, 0)

    

    return total_caps / len(s)



# Limit the category

entertainment = df[df['category'] == 'Entertainment']



titles = entertainment['title']



# Record proportion for every row

prop = []

for title in titles:

    prop.append(proportion(title))

    

entertainment['caps_prop'] = prop



plt.figure(figsize=(12,6))

g = sns.scatterplot(

    data = entertainment,

    x = 'caps_prop',

    y = 'views'

)

g.set(title='Caps_prop versus Views')

plt.show()