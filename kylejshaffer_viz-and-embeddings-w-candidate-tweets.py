# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

plt.style.use('ggplot')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

% matplotlib inline
df = pd.read_csv('../input/tweets.csv')

print(df.shape)

df.head()
df['dummy_count'] = 1
df['time_decoded'] = pd.to_datetime(df.time)

df['time_decoded'] = df.time_decoded.map(lambda x: x.strftime('%Y-%m-%d'))

df[['time', 'time_decoded']].head()
df.handle.value_counts().plot(kind='bar', color='forestgreen')
grouped = df.groupby(['time_decoded', 'handle']).dummy_count.sum().reset_index()

grouped.head()
# Let's look at tweets over time

# VERY interesting to see the influx of Clinton tweets starting around July '16

grouped.pivot(index='time_decoded', columns='handle', values='dummy_count').plot(alpha=0.5, figsize=(9, 5), color=['steelblue', 'darkred'], title='Distribution of Tweets Over Time')
import nltk, string

from gensim.models import Word2Vec

from nltk.corpus import stopwords



trump_tweets = [t.strip().lower().split() for t in df[df.handle=='realDonaldTrump'].text.tolist()]

clinton_tweets = [t.strip().lower().split() for t in df[df.handle=='HillaryClinton'].text.tolist()]



# Cleaning

punct = string.punctuation

remove = stopwords.words('english')

remove.extend(['&amp', 'amp', '&amp;', 'realDonaldTrump', 'HillaryClinton'])

# Remove punctuation

trump_tweets = [[''.join(ch for ch in w if not ch in punct) for w in t] for t in trump_tweets]

clinton_tweets = [[''.join(ch for ch in w if not ch in punct) for w in t] for t in clinton_tweets]



# Remove stopwords

trump_tweets = [[w for w in t if not w in remove] for t in trump_tweets]

clinton_tweets = [[w for w in t if not w in remove] for t in clinton_tweets]



# Prefixing

trump_tweets = [['trump_' + w for w in t] for t in trump_tweets]

clinton_tweets = [['clinton_' + w for w in t] for t in clinton_tweets]

prefixed_tweets = trump_tweets + clinton_tweets



# Fit W2V model on cleaned up tweets

w2v = Word2Vec(prefixed_tweets, window=3, min_count=5)
w2v.most_similar('clinton_trump')
w2v.most_similar('trump_clinton')
flattened_tokens = [word for tweet in prefixed_tweets for word in tweet]

freq_dist = nltk.FreqDist(flattened_tokens)

for word, freq in freq_dist.most_common(20):

    print("{}: {}".format(word, freq))
from sklearn.manifold import TSNE



def plot_with_labels(low_dim_embs, labels, colors):

  assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"

  plt.figure(figsize=(9, 9))  #in inches

  for i, label in enumerate(labels):

    x, y = low_dim_embs[i,:]

    plt.scatter(x, y, color=colors[i])

    plt.annotate(label,

                 fontsize=6,

                 xy=(x, y),

                 xytext=(5, 2),

                 textcoords='offset points',

                 ha='right',

                 va='bottom')



top_words = [word for word, freq, in freq_dist.most_common(250)]

most_common_embeddings = np.zeros((len(top_words), w2v.vector_size))



colors = []

for w in top_words:

    if 'trump_' in w:

        colors.append('r')

    elif 'clinton_' in w:

        colors.append('b')



for idx, w in enumerate(top_words):

    most_common_embeddings[idx] = w2v[w]



labels = [w.replace('trump_', '') for w in top_words]

labels = [w.replace('clinton_', '') for w in labels]



tsne = TSNE(perplexity=30, n_components=2, n_iter=1000)

low_dim_embeds = tsne.fit_transform(most_common_embeddings)

plot_with_labels(low_dim_embeds, labels, colors)