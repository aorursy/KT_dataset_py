# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt 

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

test_df = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
train_df[~train_df.keyword.isnull()].head()
train_df[train_df.target == 1 & ~train_df.keyword.isnull()].keyword.nunique()
df = train_df[train_df.target == 1 & ~train_df.keyword.isnull()].keyword.value_counts().to_frame('count').reset_index()

df.columns = ['keyword', 'count']

df.head(10)
train_df[train_df["target"] == 0]["text"].values[1:5]
train_df[train_df["target"] == 1]["text"].values[1:5]
sns.countplot(x='target', data=train_df)
keywords = dict(train_df.keyword.value_counts())

wordcloud = WordCloud(width=800, height=400,background_color="white").generate_from_frequencies(keywords)

plt.figure(figsize=[10,8])

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()
keywords = dict(test_df.keyword.value_counts())

wordcloud = WordCloud(width=800, height=400,background_color="white").generate_from_frequencies(keywords)

plt.figure(figsize=[10,8])

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()
sns.distplot(train_df.text.str.len(), bins=20, kde=False, rug=True);
print("Avg tweet length of no disaster {0}" .format(train_df[train_df.target == 0].text.str.len().mean()))

print("Avg tweet length of disaster {0}" .format(train_df[train_df.target == 1].text.str.len().mean()))
def word_length(tweet):

    return len(tweet.split(' '))
train_df['len'] = train_df.text.apply(word_length)
print("Avg word length of no disaster {0}" .format(train_df[train_df.target == 0].len.mean()))

print("Avg word length of disaster {0}" .format(train_df[train_df.target == 1].len.mean()))
print("Avg word length of a tweet {0}".format(train_df.len.mean()))
sns.distplot(train_df.len, bins=20, kde=False, rug=True);
train_df.text.values[36] #Positive ablaze
train_df.text.values[46] 