import numpy as np

import pandas as pd

import json

import matplotlib.pyplot as plt

import matplotlib.markers

import os

import seaborn as sns

import pprint

import string

import re



from tqdm import tqdm



plt.style.use('ggplot')

tqdm.pandas()
df_train = pd.read_csv('../input/train.csv')
df_train.shape
df_train.info()
df_train.head()
# import the json file to view the categories



with open('../input/categories.json', 'rb') as handle:

    cat_details = json.load(handle)
pprint.pprint(cat_details)
category_mapper = {}

product_type_mapper = {}



for category in cat_details.keys():

    for key, value in cat_details[category].items():

        category_mapper[value] = key

        product_type_mapper[value] = category
# Display category mapper



category_mapper
# Display product mapper



product_type_mapper
# Apply the mapper to get new columns - category_type and product_type



df_train['Category_type'] = df_train['Category'].map(category_mapper)

df_train['Product_type'] = df_train['Category'].map(product_type_mapper)
plt.figure(figsize=(12,6))

plot = sns.countplot(x='Product_type', data=df_train)

plt.title('Product Type %', fontsize=20)

ax = plot.axes



for p in ax.patches:

    ax.annotate(f'{p.get_height() * 100 / df_train.shape[0]:.2f}%',

                (p.get_x() + p.get_width() / 2., p.get_height()), 

                ha='center', 

                va='center', 

                fontsize=11, 

                color='black',

                xytext=(0,7), 

                textcoords='offset points')
for product in cat_details.keys():

    plt.figure(figsize=(20,6))

    plot = sns.countplot(x='Category_type', 

                         data = df_train.loc[df_train['Product_type'] == product, :], 

                         order = df_train.loc[df_train['Product_type'] == product, 'Category_type'].value_counts().index)

    plt.xticks(rotation=90)

    plt.title(f'Category breakdown ({product})', fontsize=20)

    ax = plot.axes



    for p in ax.patches:

        ax.annotate(f'{p.get_height() * 100 / df_train.shape[0]:.2f}%',

                    (p.get_x() + p.get_width() / 2., p.get_height()), 

                    ha='center', 

                    va='center', 

                    fontsize=11, 

                    color='black',

                    xytext=(0,7), 

                    textcoords='offset points')

    plt.show()
from nltk import FreqDist

from nltk.corpus import stopwords

from wordcloud import WordCloud
df_train.head()
def preprocessing(titles_array):

    

    processed_array = []

    

    for title in titles_array:

        

        # remove digits and other non-alphabets symbols with space (i.e. keep only alphabets and whitespaces)

        processed_title = re.sub('[^a-zA-Z ]', '', title.lower())

        words = processed_title.split()

        

        # remove words that have length of 1

        processed_array.append([word for word in words if len(word) > 1])

    

    return processed_array
def get_freqdist_wc(titles, product_type, num_words=30):

    

    freq_dist = FreqDist([word for title in titles for word in title])

    wordcloud = WordCloud(background_color='White').generate_from_frequencies(freq_dist)

    

    plt.figure(figsize=(22,6))

    plt.subplot2grid((1,5),(0,0),colspan=2)

    plt.imshow(wordcloud,interpolation='bilinear')

    plt.axis('off')



    plt.subplot2grid((1,5),(0,2),colspan=3)

    plt.title(f'Frequency Distribution ({product_type}, Top {num_words})', fontsize=20)

    freq_dist.plot(num_words, marker='|', markersize=20)



    plt.tight_layout()

    plt.show()
mobile_titles = df_train.loc[df_train['Product_type'] == 'Mobile','title'].values

fashion_titles = df_train.loc[df_train['Product_type'] == 'Fashion','title'].values

beauty_titles = df_train.loc[df_train['Product_type'] == 'Beauty','title'].values



mobile_titles_p = preprocessing(mobile_titles)

fashion_titles_p = preprocessing(fashion_titles)

beauty_titles_p = preprocessing(beauty_titles)
get_freqdist_wc(mobile_titles_p, 'Mobile')
get_freqdist_wc(fashion_titles_p, 'Fashion')
get_freqdist_wc(beauty_titles_p, 'Beauty')
def process_and_plot(cat_type, num_words = 10):

    titles = df_train.loc[df_train['Category_type'] == cat_type,'title'].values

    processed_titles = preprocessing(titles)

    print(f'{cat_type}\'s total counts:\t {len(titles)}')

    print(f'{len(titles) * 100/ df_train.shape[0]:.2f}% of the training set.')

    get_freqdist_wc(processed_titles, cat_type, num_words)

    
for category in tqdm(list(category_mapper.values())):

    process_and_plot(category)