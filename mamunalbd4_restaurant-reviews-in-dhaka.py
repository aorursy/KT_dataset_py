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
import seaborn as sns

import matplotlib.pyplot as plt
df = pd.read_csv('/kaggle/input/restaurant-reviews-in-dhaka-bangladesh/reviews.csv')
df.head()
df.isnull().sum()
df.head()
pd.set_option('display.max_colwidth',150)
df.head()
import string

import nltk

import re
def lower_caes(txt):

    return txt.lower()
def remove_punctuation(txt):

    txt_clean = "".join([c for c in txt if c not in string.punctuation])

    return txt_clean
df['lower_case'] = df['Review Text'].apply(lambda x: lower_caes(x))
df.head()
df = df[['lower_case']]
df.head()
df['review'] = df['lower_case'].apply(lambda x: remove_punctuation(x))
df.head()
df = df[['review']]
df.head()
from nltk.sentiment import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()
sid.polarity_scores(df['review'].iloc[1])
df['scores'] = df['review'].apply(lambda x: sid.polarity_scores(x))
df.head()
df['compound'] = df['scores'].apply(lambda x: x['compound'])
df.head()
df['tag'] = df['compound'].apply(lambda x: 'pos' if x>0.15 else 'neg')
df.head()
df['tag'].value_counts()
df = df[['review', 'tag']]
df.head()
sns.countplot(x = 'tag', data = df)