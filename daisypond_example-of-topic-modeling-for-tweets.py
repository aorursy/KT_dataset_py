!pip install mglearn

import re

import numpy as np

import pandas as pd

import sklearn as sk

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.decomposition import LatentDirichletAllocation

import mglearn

import matplotlib.pyplot as plt
tweet_df = pd.read_csv('../input/auspol2019.csv')
tweet_df.head(5)
tweet_df['full_text2'] = tweet_df['full_text'].map(lambda x: re.sub('[,\.!?]', '', x))                                       # remove ,.!?

tweet_df['full_text2'] = tweet_df['full_text2'].map(lambda x: re.sub('#[A-Za-z0-9]+', '', x))                       # remove hashtag

tweet_df['full_text2'] = tweet_df['full_text2'].map(lambda x: re.sub('https://t.co/[A-Za-z0-9]+', '', x))  # remove link





vect = CountVectorizer(max_features=100, max_df=.15, stop_words='english')

X = vect.fit_transform(tweet_df["full_text2"])



topics =2

lda = LatentDirichletAllocation(n_components = topics, learning_method="batch", max_iter=5, random_state=0)

document_topics = lda.fit_transform(X)

sorting = np.argsort(lda.components_, axis=1)[:, ::-1]

feature_names = np.array(vect.get_feature_names())

mglearn.tools.print_topics(topics=range(topics), feature_names=feature_names, sorting=sorting, topics_per_chunk=5, n_words=10)
topic_df = pd.DataFrame(document_topics, columns=['topic0', 'topic1'])

tweet_df = pd.concat([tweet_df.reset_index(drop=True), topic_df.reset_index(drop=True)],axis =1)

tweet_df["topic_diff"] = tweet_df['topic1']-tweet_df['topic0']

tweet_df.head()
plt.hist(tweet_df['topic_diff'], 20)

plt.xlabel("<--- Scott Morrison    Bill Shorten -->        ")

plt.show()
tweet_df.describe(percentiles=[.95, .99, .999])
th99 = 1224



tweet0 = tweet_df.query('topic_diff < -0.75 & favorite_count > ' +str(th99))



tw0= tweet0['full_text'].values



for tw in tw0:

    print('\n' + tw.replace('\n', ' '))

tweet1 = tweet_df.query('topic_diff > 0.75 & favorite_count > ' + str(th99))

tw1= tweet1['full_text'].values



for tw in tw1:

    print('\n' + tw.replace('\n', ' '))