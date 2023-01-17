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
!pip install tweet-preprocessor
import preprocessor as p
train_df = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')
train_df.count()
train_df = train_df.dropna()

train_df = train_df.drop_duplicates()
train_df.count()
train_df.head()
def preprocess_tweet(row):

    text = row['text']

    text = p.clean(text)

    return text
train_df['text'] = train_df.apply(preprocess_tweet, axis=1)
train_df.head()
from gensim.parsing.preprocessing import remove_stopwords
def stopword_removal(row):

    text = row['text']

    text = remove_stopwords(text)

    return text
train_df['text'] = train_df.apply(stopword_removal, axis=1)
train_df.head()
train_df['text'] = train_df['text'].str.lower().str.replace('[^\w\s]',' ').str.replace('\s\s+', ' ')
train_df.head()