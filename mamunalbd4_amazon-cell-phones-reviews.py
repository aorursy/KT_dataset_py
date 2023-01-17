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

%matplotlib inline
item = pd.read_csv('/kaggle/input/amazon-cell-phones-reviews/20190928-items.csv')

item.head()
review = pd.read_csv('/kaggle/input/amazon-cell-phones-reviews/20190928-reviews.csv')

review.head()
df = pd.merge(item, review, how='right', on='asin')

df.head()
df = df[['asin', 'brand', 'rating_x', 'totalReviews', 'rating_y', 'title_y', 'body', 'helpfulVotes']]

df.head()
df.isnull().sum()
del df['helpfulVotes']
df.isnull().sum()
df.dropna(inplace=True)
df.isnull().sum()
len(df['asin'])
plt.figure(figsize=(20,12))

sns.countplot(x = 'brand', data =df)
plt.figure(figsize=(20,8))

sns.boxplot(x='brand', y= 'totalReviews', data =df)
df.head()
df['body'] = df['body'].str.lower()

df['title_y'] = df['title_y'].str.lower()
import re

import string

import nltk
def remove_punctuation(text):

    no_punct = ''.join([c for c in text if c not in string.punctuation])

    return no_punct
df['body'] = df['body'].apply(lambda x: remove_punctuation(x))

df['title_y'] = df['title_y'].apply(lambda x: remove_punctuation(x))
df.head()
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()
sid.polarity_scores(df['body'].iloc[11])
df['title_score'] = df['title_y'].apply(lambda x: sid.polarity_scores(x))
df.head()
df['title_compound'] = df['title_score'].apply(lambda x: x['compound'])
df.head()
df['title_com_review'] = df['title_compound'].apply(lambda x: 'pos' if x>= 0.05 else 'neg')
df.head()
df['title_com_review'].value_counts()
df['body_score'] = df['body'].apply(lambda x: sid.polarity_scores(x))
df.head()
df['body_compound'] = df['body_score'].apply(lambda x: x['compound'])
df['body_com_review'] = df['body_compound'].apply(lambda x: 'pos' if x>= 0.05 else 'neg')
df.head()
plt.figure(figsize=(18,8))

sns.countplot(x = 'brand', hue = 'title_com_review', data = df)

plt.xlabel('Title name positive and Negative', fontsize = 18)

plt.ylabel('Count', fontsize = 18)

plt.title('Title sentiment analysis', fontsize = 24)
plt.figure(figsize=(18,8))

sns.countplot(x = 'brand', hue = 'body_com_review', data = df)

plt.xlabel('Comments positive and Negative', fontsize = 18)

plt.ylabel('Count', fontsize = 18)

plt.title('Comments sentiment analysis', fontsize = 24)
plt.figure(figsize=(18,8))

sns.barplot(x = 'brand', y = 'totalReviews', hue = 'body_com_review', data = df)
plt.figure(figsize=(18,8))

sns.violinplot(x = 'brand', y = 'totalReviews', hue = 'body_com_review', data = df)
plt.figure(figsize=(12,6))

sns.countplot(x = 'title_com_review', data = df)

plt.xlabel('Total Title positive and Negative', fontsize = 18)

plt.ylabel('Total Count', fontsize = 18)
plt.figure(figsize=(12,6))

sns.countplot(x = 'body_com_review', data = df)

plt.xlabel('Total comments positive and Neg')
df['body_com_review'].value_counts()
df.head()
plt.figure(figsize=(20,6))

sns.pointplot(x = 'brand', y = 'rating_y', hue = 'body_com_review', data = df)

plt.title('Comments sentiment analysis vs rating_y', fontsize = 24)

plt.figure(figsize=(20,6))

sns.pointplot(x = 'brand', y = 'rating_y', hue = 'title_com_review', data = df)
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
print(classification_report(df['title_com_review'], df['body_com_review']))