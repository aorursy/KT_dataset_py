# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import io

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from nltk import ngrams

from nltk import sentiment

import re

import random

from collections import defaultdict

nltk.download('vader_lexicon')



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
ye = pd.DataFrame({'lyrics': io.open('../input/kanyewestverses/kanye_verses.txt', 'r', encoding='ascii', errors='ignore').read().split('\n\n')})
ye.loc[0, 'lyrics']
# characters, words, lines

ye['#characters'] = ye.lyrics.str.len()

ye['#words'] = ye.lyrics.str.split().str.len()

ye['#lines'] = ye.lyrics.str.split('\n').str.len()

ye['#uniq_words'] = ye.lyrics.apply(lambda x: len(set(x.split())))

ye['lexical_density'] = ye['#uniq_words'] / ye['#words']
ye.head()
ye.hist(sharey=True, layout=(2, 3), figsize=(15, 8));


# Word length distribution

pd.Series(len(x) for x in ' '.join(ye.lyrics).split()).value_counts().sort_index().plot(kind='bar', figsize=(12, 3))
#top words

pd.Series(' '.join(ye.lyrics).lower().split()).value_counts()[:20][::-1].plot(kind='barh')
# top long words

pd.Series([w for w in ' '.join(ye.lyrics).lower().split() if len(w) > 5]).value_counts()[:20][::-1].plot(kind='barh')
def get_ngrams_from_series(series, n=2):

    # using nltk.ngrams

    lines = ' '.join(series).lower().split('\n')

    lgrams = [ngrams(l.split(), n) for l in lines]

    grams = [[' '.join(g) for g in list(lg)] for lg in lgrams]

    return [item for sublist in grams for item in sublist]
#Top bi-grams

pd.Series(get_ngrams_from_series(ye.lyrics, 2)).value_counts()[:20][::-1].plot(kind='barh')
pd.Series(get_ngrams_from_series(ye.lyrics, 3)).value_counts()[:20][::-1].plot(kind='barh')
pd.Series(get_ngrams_from_series(ye.lyrics, 4)).value_counts()[:20][::-1].plot(kind='barh')
senti_analyze = sentiment.vader.SentimentIntensityAnalyzer()

senti_analyze.polarity_scores(ye.lyrics[0])



ye['sentiment_score'] = pd.DataFrame(ye.lyrics.apply(senti_analyze.polarity_scores).tolist())['compound']

ye['sentiment'] = pd.cut(ye['sentiment_score'], [-np.inf, -0.35, 0.35, np.inf], labels=['negative', 'neutral', 'positive'])
ye
ye.lyrics[10]
ye.lyrics[15]
ye.lyrics[23]
ye[['sentiment_score']].hist(bins=25)
# Songs with lower lexical density tend to have strong sentiments

ye.plot.scatter(x='sentiment_score', y='lexical_density', s=ye['#characters']/20,

                c=np.where(ye['lexical_density'].le(0.55), '#e41a1c', '#4c72b0'),

                figsize=(15, 6))
# Song themes via Simplistic topic modelling



from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.decomposition import NMF



no_topics = 5

no_features = 50

tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')

tfidf = tfidf_vectorizer.fit_transform(ye.lyrics)

tfidf_feature_names = tfidf_vectorizer.get_feature_names()

nmf = NMF(n_components=no_topics, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(tfidf)



def get_topics(model, feature_names, no_topwords):

    for topic_id, topic in enumerate(model.components_):

        print('topic %d:' % (topic_id))

        print(' '.join([feature_names[i] for i in topic.argsort()[:-no_topwords-1:-1]]))



s = pd.DataFrame(nmf.transform(tfidf)).idxmax(1)
# NMP topics

get_topics(nmf, tfidf_feature_names, 30)
# Top n-grams from the topics

topics = set(s)

fig, axs = plt.subplots(figsize=(18, 6), ncols=len(topics))

for i, v in enumerate(topics):

    dfsm = ye.loc[s.eq(v), 'lyrics']

    ngram = pd.Series(get_ngrams_from_series(dfsm, 3)).value_counts()[:20][::-1]

    ngram.plot(kind='barh', ax=axs[i], title='Topic {} - {} lyrics'.format(v, s.eq(v).sum()))

plt.tight_layout()

ye['topic'] = s.astype(str).radd('Topic ')
fig, axs = plt.subplots(figsize=(15, 6))

sns.swarmplot(x='topic', y='sentiment_score', data=ye)
# Machine generated lyrics using Markov

class MarkovRachaita:

    def __init__(self, corpus='', order=2, length=8):

        self.order = order

        self.length = length

        self.words = re.findall("[a-z']+", corpus.lower())

        self.states = defaultdict(list)



        for i in range(len(self.words) - self.order):

            self.states[tuple(self.words[i:i + self.order])].append(self.words[i + order])



    def gen_sentence(self, length=8, startswith=None):

        terms = None

        if startswith:

            start_seed = [x for x in self.states.keys() if startswith in x]

            if start_seed:

                terms = list(start_seed[0])

        if terms is None:

            start_seed = random.randint(0, len(self.words) - self.order)

            terms = self.words[start_seed:start_seed + self.order]



        for _ in range(length):

            terms.append(random.choice(self.states[tuple(terms[-self.order:])]))



        return ' '.join(terms)



    def gen_song(self, lines=10, length=8, length_range=None, startswith=None):

        song = []

        if startswith:

            song.append(self.gen_sentence(length=length, startswith=startswith))

            lines -= 1

        for _ in range(lines):

            sent_len = random.randint(*length_range) if length_range else length

            song.append(self.gen_sentence(length=sent_len))

        return '\n'.join(song)
kanyai = MarkovRachaita(corpus=' '.join(ye.lyrics))

kai = kanyai.gen_song(lines=10, length_range=[5, 10])

kai = kai.splitlines()

kai
kanyai = MarkovRachaita(corpus=' '.join(ye.lyrics))

kai2 = kanyai.gen_song(lines=10, length_range=[5, 10])

kai2 = kai2.splitlines()

kai2