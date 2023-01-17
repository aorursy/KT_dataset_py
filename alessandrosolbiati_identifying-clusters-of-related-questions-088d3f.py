import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import nltk
from multiprocessing import Pool
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import re
from itertools import chain
from collections import Counter
import pickle
import scipy.io as scio
from sklearn.decomposition import TruncatedSVD
import scipy.spatial.distance as distance
import scipy.cluster.hierarchy as hierarchy
from scipy.stats import pearsonr
dat = pd.read_csv("../input/Questions.csv", encoding='latin1')
dat['Title'].fillna("None", inplace=True)
dat['Score'].fillna(0, inplace=True)
dat.iloc(0)[0]
# select a sample - results will improve without sampling in tf-idf caluculations, but due to
# Kaggle kernel memory limit we have to make a compromise here.
selected_ids = np.random.choice(range(dat.shape[0]), 10000, replace=False)
sample = dat.loc[selected_ids, :]
sample.shape
sample.head()
def purify_string(html):
    """
    this will apply to the sample
    """
    return re.sub('(\r\n)+|\r+|\n+', " ", re.sub('<[^<]+?>', '', html))
corpus = sample.ix[:, 'Body'].apply(purify_string)
def combine_title_body(tnb):
    return tnb[0] + " " + tnb[1]
p = Pool(8)
combined_corpus = p.map(combine_title_body, zip(dat['Title'], corpus))
p.close()
combined_corpus[:2]
lem = WordNetLemmatizer()
def cond_tokenize(t):
    if t is None:
        return []
    else:
        return [lem.lemmatize(w.lower()) for w in word_tokenize(t)]

p = Pool(8)
tokens = list(p.imap(cond_tokenize, combined_corpus))
p.close()
# stops = stopwords.words('english')
pure_tokens = [" ".join(sent) for sent in tokens]
print(tokens[0]) # this are the single lemmatized and stemmed tokens
print(pure_tokens[0]) # these are the tokens combined in original form
vectorizer = TfidfVectorizer(min_df=1, max_features=2000, stop_words='english', ngram_range=[1, 1], sublinear_tf=True)
tfidf = vectorizer.fit_transform(pure_tokens) # this is the vector matrix of the tfidf
idfs = pd.DataFrame([[v, k] for k, v in vectorizer.vocabulary_.items()], columns=['id', 'word']).sort_values('id')
idfs['idf'] = vectorizer.idf_
 # *this is the IDFS vector that can be used to examine how the TFIDF worked*
print(idfs.sort_values('idf').head(10))
tsvd = TruncatedSVD(n_components=500) # TODO this n_components=500 is a hyperparameter, look into it
transformed = tsvd.fit_transform(tfidf)
np.sum(tsvd.explained_variance_ratio_)
transformed.shape
# calculate pairwise cosine distance
D = distance.pdist(transformed, 'cosine')
# hierarchical clustering - tree calculation
L = hierarchy.linkage(D)
# mean distance between clusters
np.mean(D)
# split clusters by criterion. Here 0.71 is used as the inconsistency criterion. Adjust the
# number to change cluster sizes
# TODO : this is the second hyperparameters, look into it
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.fcluster.html#scipy.cluster.hierarchy.fcluster
cls = hierarchy.fcluster(L, 0.71, criterion='inconsistent')
df_cls = pd.DataFrame({'Pos': selected_ids, 'Cluster': cls})
cnts = df_cls.groupby('Cluster').size().sort_values(ascending=False)
cnts.sort_values(ascending=False).head()
# add clusters to question data
bc = pd.concat([sample, df_cls.set_index('Pos')], axis=1)
bc.head()
# calculate cluster stats
stats = bc.groupby('Cluster')['Score'].describe().unstack()
stats.sort_values(ascending=False).head(10)
plt.figure(figsize=(12, 8))
plt.hlines([0], xmin=0, xmax=np.max(stats['count']) + 5, alpha=0.5)
plt.vlines([1], ymin=0, ymax=np.max(stats['mean']) + 50, alpha=0.5)
plt.scatter(stats['count'], stats['mean'], alpha=0.3)
plt.title("cluster mean score vs cluster size")
plt.xlabel("cluster size")
plt.ylabel("mean score")
plt.show()
bc.loc[bc['Cluster'] == cnts.index[0]][['Score', 'Title', 'Body']]
bc.loc[bc['Cluster'] == cnts.index[1]][['Score', 'Title', 'Body']]
bc.loc[bc['Cluster'] == cnts.index[2]][['Score', 'Title', 'Body']]
