# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.express as px # interactive plotting

import matplotlib.pyplot as plt



from sklearn.feature_extraction.text import TfidfVectorizer # tf-idf is a common way of turning a text into a vector of term frequencies. Texts that are semantically close will be close in this (high-dimensional) space

from sklearn.manifold import TSNE # t-SNE is a dimensionality reduction tool which is optimized to preserve distances, i.e. points that are close in the high-dimensional space should be close in the low-dimensional space and vice versa

from sklearn.cluster import DBSCAN # density based clustering method. It groups together points that are closely packed together (points with many nearby neighbors), marking as outliers points that lie alone in low-density regions (source: Wikipedia)



from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator # for drawing word clouds and also setting stop words for vectorization
# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

df = pd.read_csv('../input/CORD-19-research-challenge/metadata.csv')

df.info()
df_clean = df.loc[:,['title','abstract','doi','has_full_text']].dropna(subset=['title','abstract']).drop_duplicates('title')
df_clean.head()
ABSTRACT_STOPWORDS = set(['BACKGROUND','METHOD','METHODS','RESULT','RESULTS','CONCLUSION','CONCLUSIONS','SUPPLEMENTARY MATERIAL','AIM','CONTEXT','PARTICIPANTS','SUBJECTS','PATIENTS','ABSTRACT'])

X = TfidfVectorizer(input='content', stop_words=list(STOPWORDS.union(ABSTRACT_STOPWORDS)), max_features=200).fit_transform(df_clean.loc[:, 'abstract']).toarray()
df_clean['x_tsne'] = np.nan

df_clean['y_tsne'] = np.nan

df_clean.loc[:,['x_tsne','y_tsne']] = TSNE().fit_transform(X)
fig = px.scatter(df_clean.reset_index(), x='x_tsne', y='y_tsne', hover_name='title', hover_data=['index'], width=1000, height=1000)

fig.show()
df_clean['cluster'] = DBSCAN(min_samples=1).fit(df_clean.loc[:,['x_tsne','y_tsne']]).labels_

df_clean.cluster = df_clean.cluster.replace(dict(np.array(list(enumerate(df_clean.cluster.value_counts().sort_values(ascending=False).index)))[:,::-1])) #so that cluster 0 is the biggest, 1 is the next biggest etc.
fig = px.scatter(df_clean.loc[df_clean.cluster<200].reset_index(), x='x_tsne', y='y_tsne', hover_name='title', color='cluster', hover_data=['cluster','index'], width=1000, height=1000)

fig.show()
# Any results you write to the current directory are saved as output.

df_clean.to_csv('df_clean.csv')
TITLE_STOPWORDS = set([])
cluster_number = 187

title_wordcloud = WordCloud(width=600, height=500, stopwords=STOPWORDS.union(TITLE_STOPWORDS)).generate(' '.join(df_clean.loc[df_clean.cluster==cluster_number, 'title']))

abstract_wordcloud = WordCloud(width=600, height=500, stopwords=STOPWORDS.union(ABSTRACT_STOPWORDS)).generate(' '.join(df_clean.loc[df_clean.cluster==cluster_number, 'abstract']))

fig, axes = plt.subplots(ncols=2, figsize=(30,10))

axes[0].imshow(title_wordcloud, interpolation='bilinear')

axes[0].set_title('Titles')

axes[1].imshow(abstract_wordcloud, interpolation='bilinear')

axes[1].set_title('Abstracts')

for i in range(2):

    axes[i].set_xticks([])

    axes[i].set_yticks([])

fig.suptitle('Word cloud for %d articles in cluster %d' % ((df_clean.cluster==cluster_number).sum(), cluster_number))

plt.show()