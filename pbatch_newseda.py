import pandas as pd

import numpy as np

import os

from wordcloud import WordCloud

from collections import Counter

import matplotlib.pyplot as plt

import seaborn as sns

from nltk.corpus import stopwords

sns.set_style('whitegrid')
def bar_plot(col, n_keys=None, d=None):

    

    if d is None:

        d = Counter(df[col].dropna())

    d = {k: v for k, v in sorted(d.items(), key=lambda item: -item[1])}

    plt.figure(figsize=(20,4))

    plt.ylabel('Frequency')

    plt.xlabel(col.capitalize())

    

    if n_keys is None:

        keys = list(d.keys())

        values = list(d.values())

    else:

        keys = list(d.keys())[:n_keys]

        values = list(d.values())[:n_keys]

        keys.append('other')

        values.append(sum(list(d.values())[n_keys:]))



    plt.bar(keys, values, width=0.8)

    return None
df = pd.read_csv('/kaggle/input/coronanews/clean_news.csv')

df.head(3)
bar_plot('source')
bar_plot('publish_date', n_keys=10)
def calculate_length(text):

    try:

        return len(text.split(' '))

    except AttributeError:

        return 0



article_lengths = df['text'].apply(lambda x: calculate_length(x))

plt.figure(figsize=(20,4))

plt.xlabel('Article lengths')

plt.ylabel('Frequency')

ax = plt.hist(article_lengths, bins=25)
text = ' '.join(list(df['text'].dropna()))

bad_words = {'Getty Images', 'Getty', 'AFP', 'via', 'could', 'would', 'also', 'said'}

stop_words = set(stopwords.words('english')) | bad_words



wordcloud = WordCloud(stopwords=stop_words).generate(text)

plt.figure(figsize=(16, 8))

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.show()