# Setting package umum 

import pandas as pd

import pandas_profiling as pp

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec

import seaborn as sns

from tqdm import tqdm_notebook as tqdm

import time

import tensorflow as tf

%matplotlib inline



from matplotlib.pylab import rcParams

# For every plotting cell use this

# grid = gridspec.GridSpec(n_row,n_col)

# ax = plt.subplot(grid[i])

# fig, axes = plt.subplots()

rcParams['figure.figsize'] = [10,5]

plt.style.use('fivethirtyeight') 

sns.set_style('whitegrid')



import warnings

warnings.filterwarnings('ignore')

from tqdm import tqdm



pd.set_option('display.max_rows', 100)

pd.set_option('display.max_columns', 50)

pd.options.display.float_format = '{:.5f}'.format



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Load dataset

df_scrape = pd.read_csv('/kaggle/input/shopee-reviews/shopee_reviews.csv')

df_test = pd.read_csv('/kaggle/input/shopee-sa-cleaning/test_cleaned.csv')
### Remove unwanted rows

df_scrape = df_scrape[df_scrape['label']!='label']
### Change label to numeric

df_scrape['label'] = df_scrape['label'].astype('int')
### Change it into competition format

df_scrape = df_scrape[['text','label']]

df_scrape.columns = ['review','rating']
### Function to sampling to mimic test dataset distribution

def dataset_sampling(dataset) :

    

    df = df_scrape.copy()



    # Cut class 5 observation

    df_no_c5 = df[df['rating']!=5]

    df_c5 = df[df['rating']==5]

    n_to_sample = (len(df_no_c5)*0.35) / (1-0.35)



    df_c5_sampled = df_c5.sample(n=int(np.round(n_to_sample)))

    df = pd.concat([df_no_c5, df_c5_sampled]).sample(frac=1)

    

    # Cut class 3 observation

    df_no_c3 = df[df['rating']!=3]

    df_c3 = df[df['rating']==3]

    n_to_sample = (len(df_no_c3)*0.06) / (1-0.06)



    df_c3_sampled = df_c3.sample(n=int(np.round(n_to_sample)))

    df = pd.concat([df_no_c3, df_c3_sampled]).sample(frac=1)

    

    # Cut class 2 observation

    df_no_c2 = df[df['rating']!=2]

    df_c2 = df[df['rating']==2]

    n_to_sample = (len(df_no_c3)*0.02) / (1-0.02)



    df_c2_sampled = df_c2.sample(n=int(np.round(n_to_sample)))

    df = pd.concat([df_no_c2, df_c2_sampled]).sample(frac=1)



    return df.reset_index(drop=True)
### Sampling dataset

df_scrape = dataset_sampling(df_scrape)
### Function to quick clean

import string

import re

import emoji  



def remove_punctuation(text) :

    no_punct = ''.join([c for c in text if c not in string.punctuation])

    

    return no_punct



def encode_emoticon(text) :

    

    text = re.sub(r':\(', 'dislike', text)

    text = re.sub(r': \(\(', 'dislike', text)

    text = re.sub(r':, \(', 'dislike', text)

    text = re.sub(r':\)', 'smile', text)

    text = re.sub(r';\)', 'smile', text)

    text = re.sub(r':\)\)\)', 'smile', text)

    text = re.sub(r':\)\)\)\)\)\)', 'smile', text)

    text = re.sub(r'=\)\)\)\)', 'smile', text)

    

    return text



def quick_clean_data(dataset, var) :

    

    df = dataset.copy()

    

    # Lowercase

    df[var] = df[var].str.lower()

    

    # Strip whitespaces

    df[var] = df[var].str.strip()

    

    # Remove punctuation

    df[var] = df.apply(lambda x : remove_punctuation(x[var]), axis=1)

    

    # Remove double whitespaces

    df[var] = df.apply(lambda x : " ".join(x[var].split()), axis=1)

    

    # Change emoticon to text

    df[var] = df.apply(lambda x : encode_emoticon(x[var]), axis=1)

    

    return df
### Decode emoji

import emoji  



def find_emoji(text) :

    

    # Change emoji to text

    text = emoji.demojize(text).replace(":", " ")

    

    # Delete repeated emoji

    tokenizer = text.split()

    repeated_list = []

    

    for word in tokenizer:

        if word not in repeated_list:

            repeated_list.append(word)

    

    text = ' '.join(text for text in repeated_list)

    text = text.replace("_", " ").replace("-", " ")

    

    return text



def encode_emoji(dataset, var) :

    

    df = dataset.copy()

    

    # Get index for rows with emoji

    list_idx = []

    for idx, review in enumerate(df[var]):

        if any(char in emoji.UNICODE_EMOJI for char in review):

            list_idx.append(idx)

            

    print('Percentage of dataset with emoji :',len(list_idx)/len(df)*100)

            

    # Encode emoji

    df.loc[list_idx, var] = df.loc[list_idx, var].apply(find_emoji)

    

    return df
### Clear repeated words

# Indralin ways - dealing with bahasa text

def find_repeated_char(text) :



    text = re.sub(r'(\w)\1{2,}', r'\1', text)

    

    return text



def delete_repeated_char(dataset, var) :

    

    df = dataset.copy()

    

    # Get index for rows with repeated char

    list_idx = []

    for idx, review in enumerate(df[var]):

        if re.match(r'\w*(\w)\1+', review):

            list_idx.append(idx)

            

    print('Percentage of dataset with repeated char :',len(list_idx)/len(df)*100)

    

    # Delete repeated char

    df[var] = df[var].apply(find_repeated_char)

    

    return df
### Load english stop words

from nltk.corpus import stopwords



def remove_stop_words(text) :

    

    # List of stop words

    en_stop_words = stopwords.words('english')

    

    # Remove stop words 

    text = ' '.join([c for c in text.split() if c not in en_stop_words])    

    

    return text
### Decode unicode

from unicodedata import normalize



def unicode_char(text) :

    text = normalize('NFD', text).encode('ascii', 'ignore')

    text = text.decode('UTF-8')

    

    return text
### Remove duplicated row

def remove_duplicates(dataset, var) :

    

    df = dataset.copy()

    

    # Remove rows with inconsistent rating

    df_mean_rating = df.groupby(var).mean().reset_index()

    df_problem_review = df_mean_rating[df_mean_rating['rating'] % 1 != 0]

    review_to_exclude = list(df_problem_review[var])

    df = df[~df[var].isin(review_to_exclude)]

    

    # Remove duplicate row with consistent rating

    df = df.drop_duplicates(subset=[var])

    

    return df
### Compile all preprocessing

from tqdm._tqdm_notebook import tqdm_notebook

tqdm_notebook.pandas()



def compile_cleaning(df, var) :

    

    df = quick_clean_data(df, var)

    df = encode_emoji(df, var)

    df = delete_repeated_char(df, var)

    df[var] = df.progress_apply(lambda x : remove_stop_words(x[var]), axis=1)

    df[var] = df.progress_apply(lambda x : unicode_char(x[var]), axis=1)

    df = remove_duplicates(df, var)

    

    return df



df_scrape = compile_cleaning(df_scrape, 'review')

### Save dataset

df_scrape.to_csv('train_cleaned_scraped.csv', index=False)

df_test.to_csv('test_cleaned.csv', index=False)