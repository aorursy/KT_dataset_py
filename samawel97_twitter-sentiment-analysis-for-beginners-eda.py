import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

pd.set_option('display.max_columns', 50)

pd.set_option('display.max_rows', 150)

import os

import gc

gc.enable()

import time

import warnings

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt

import seaborn as sns

from tqdm import tqdm

from collections import Counter

import ast





from sklearn.feature_extraction.text import CountVectorizer

from textblob import TextBlob

import scipy.stats as stats



import seaborn as sns

from sklearn.manifold import TSNE

from sklearn.decomposition import PCA, TruncatedSVD

import matplotlib.patches as mpatches

import time



import seaborn as sns #for making plots

import matplotlib.pyplot as plt # for plotting

import os  # for os commands

from sklearn.manifold import TSNE

## Common Variables for Notebook 

ROOT = '/kaggle/input/nlp-getting-started/'



## load the data 

df_train = pd.read_csv(ROOT+'train.csv')

df_test = pd.read_csv(ROOT+'test.csv')

df_sub = pd.read_csv(ROOT+'sample_submission.csv')
#Looking data format and types

print(df_train.info())

print(df_test.info())

print(df_sub.info())
#Some Statistics

df_train.describe()
#Take a look at the data

df_train.head()
target = df_train['target']

sns.set_style('whitegrid')

plt.figure(figsize=(3,5))

sns.countplot(target)
df_train["text"].head()
#To check the text content we can use a list

df_train["text"].tolist()[:5]
t = df_train["text"].to_list()

for i in range(5):

    print('Tweet Number '+str(i+1)+': '+t[i])
l = df_train["location"].to_list()

print('There is '+ str(len(set(l)))+ ' different loction')
df_train['location'].value_counts().head(n=20)
# Plotting a bar graph of the number of tweets in each location, for the first ten locations listed

# in the column 'location'

location_count  = df_train['location'].value_counts()[:10,]

plt.figure(figsize=(10,5))

sns.barplot(location_count.index, location_count.values, alpha=0.9)

plt.title('Top 10 locations')

plt.ylabel('Number of Occurrences', fontsize=12)

plt.xlabel('location', fontsize=12)

plt.show()
# Plotting a bar graph of the number of tweets in each location, for the first ten locations listed

# in the column 'location'

location_count  = df_test['location'].value_counts()[:10,]

plt.figure(figsize=(10,5))

sns.barplot(location_count.index, location_count.values, alpha=0.8)

plt.title('Top 10 locations')

plt.ylabel('Number of Occurrences', fontsize=12)

plt.xlabel('location', fontsize=12)

plt.show()
df = df_train[df_train['location'].notnull()]
from sklearn import preprocessing

le = preprocessing.LabelEncoder()

df['location'] = le.fit_transform(df.location.values)
# tsne code from this great kernel: https://www.kaggle.com/janiobachmann/credit-fraud-dealing-with-imbalanced-datasets

# New_df is from the random undersample data (fewer instances)

X = df['location']

y = df['target']





# T-SNE Implementation

t0 = time.time()

X_reduced_tsne = TSNE(n_components=2, random_state=42).fit_transform(X.values.reshape(-1, 1))

t1 = time.time()

print("T-SNE took {:.2} s".format(t1 - t0))

f, (ax1) = plt.subplots(1, 1, figsize=(12,8))

# labels = ['Not Fake', 'Fake']

f.suptitle('Clusters using Dimensionality Reduction', fontsize=14)





blue_patch = mpatches.Patch(color='#0A0AFF', label='Not Fake')

red_patch = mpatches.Patch(color='#AF0000', label='Fake')





# t-SNE scatter plot

ax1.scatter(X_reduced_tsne[:,0], X_reduced_tsne[:,1], c=(y == 0), cmap='coolwarm', label='Not Fake', linewidths=2)

ax1.scatter(X_reduced_tsne[:,0], X_reduced_tsne[:,1], c=(y == 1), cmap='coolwarm', label='Fake', linewidths=2)

ax1.set_title('t-SNE', fontsize=14)



ax1.grid(True)



ax1.legend(handles=[blue_patch, red_patch])





plt.show()
df_train['keyword'].value_counts().head(n=20)
# Plotting a bar graph of the number of tweets in each keyword, for the first ten keywords listed

keyword_count  = df_train['keyword'].value_counts()[:10,]

plt.figure(figsize=(12,5))

sns.barplot(keyword_count.index, keyword_count.values, alpha=0.9)

plt.title('Top 10 keywords')

plt.ylabel('Number of Occurrences', fontsize=12)

plt.xlabel('keyword', fontsize=12)

plt.show()
df_test['keyword'].value_counts().head(n=20)
# Plotting a bar graph of the number of tweets in each keyword, for the first ten keywords listed

keyword_count  = df_test['keyword'].value_counts()[:10,]

plt.figure(figsize=(12,5))

sns.barplot(keyword_count.index, keyword_count.values, alpha=0.9)

plt.title('Top 10 keywords')

plt.ylabel('Number of Occurrences', fontsize=12)

plt.xlabel('keyword', fontsize=12)

plt.show()
keyword_train  = list(set(df_train['keyword']))

keyword_test  = list(set(df_test['keyword']))



print(len(keyword_train))

print(len(keyword_test))
def intersection(lst1, lst2): 

    lst3 = [value for value in lst1 if value in lst2] 

    return lst3 



len(intersection(keyword_train, keyword_test))
df = df_train[df_train['keyword'].notnull()]

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

df['keyword'] = le.fit_transform(df.keyword.values)
# tsne code from this great kernel: https://www.kaggle.com/janiobachmann/credit-fraud-dealing-with-imbalanced-datasets

# New_df is from the random undersample data (fewer instances)

X = df['keyword']

y = df['target']





# T-SNE Implementation

t0 = time.time()

X_reduced_tsne = TSNE(n_components=2, random_state=42).fit_transform(X.values.reshape(-1, 1))

t1 = time.time()

print("T-SNE took {:.2} s".format(t1 - t0))
f, (ax1) = plt.subplots(1, 1, figsize=(12,8))

# labels = ['Not Fake', 'Fake']

f.suptitle('Clusters using Dimensionality Reduction', fontsize=14)



X_reduced_tsne

blue_patch = mpatches.Patch(color='#0A0AFF', label='Not Fake')

red_patch = mpatches.Patch(color='#AF0000', label='Fake')





# t-SNE scatter plot

ax1.scatter(X_reduced_tsne[:,0], X_reduced_tsne[:,1], c=(y == 0), cmap='coolwarm', label='Not Fake', linewidths=2)

ax1.scatter(X_reduced_tsne[:,0], X_reduced_tsne[:,1], c=(y == 1), cmap='coolwarm', label='Fake', linewidths=2)

ax1.set_title('t-SNE', fontsize=14)



ax1.grid(True)



ax1.legend(handles=[blue_patch, red_patch])





plt.show()
df_train["text"]
len(set(df_train['text']))
df_test['text']
len(set(df_test['text']))