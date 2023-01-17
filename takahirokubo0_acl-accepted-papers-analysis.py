import numpy as np
import pandas as pd

papers = pd.read_csv("../input/acl_papers.csv", delimiter="\t")
papers.head(5)
from sklearn.feature_extraction.text import CountVectorizer


year_counts = {}
all_papers = pd.concat([papers["title"][papers["year"] == y] for y in [2016, 2017, 2018]], ignore_index=True)
vectorizer = CountVectorizer(stop_words="english", max_df=0.8, min_df=7, ngram_range=(1, 2))
vectorizer.fit(all_papers)

words = vectorizer.get_feature_names()
for y in [2016, 2017, 2018]:
    vectors = vectorizer.transform(papers["title"][papers["year"] == y])
    counts = vectors.toarray().sum(axis=0)
    word_counts = {}
    for i in range(len(words)):
        word_counts[words[i]] = counts[i]
    year_counts[y] = word_counts

year_count_df = pd.DataFrame(year_counts)
year_count_df.head(5)
def show_rate(year_freq_df, max_limit=0.5, min_limit=0):
    num_columns = year_freq_df.shape[1]
    total = np.repeat(year_freq_df.sum(axis=1).values.reshape(-1, 1), num_columns, axis=1)
    year_freq_df_rate = (year_freq_df / total)
    first_index = year_freq_df.columns[0]
    year_freq_df_rate.sort_values(by=first_index, inplace=True)
    limited = year_freq_df_rate[(year_freq_df_rate.max(axis=1) > max_limit) & (year_freq_df_rate.min(axis=1) > min_limit)]
    limited.plot.barh(stacked=True, figsize=(8, 12))
    return limited

year_count_rate_df = show_rate(year_count_df)
from sklearn.feature_extraction.text import TfidfVectorizer


year_idfs = {}
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.8, min_df=7, ngram_range=(1, 2))
vectorizer.fit(all_papers)

words = vectorizer.get_feature_names()
for y in [2016, 2017, 2018]:
    vectors = vectorizer.transform(papers["title"][papers["year"] == y])
    idfs = vectors.toarray().mean(axis=0)
    word_idfs = {}
    for i in range(len(words)):
        word_idfs[words[i]] = idfs[i]
    year_idfs[y] = word_idfs

year_idf_df = pd.DataFrame(year_idfs)
year_idf_df.head(5)
year_idf_rate_df = show_rate(year_idf_df)
from sklearn.feature_extraction.text import TfidfVectorizer


acl_2018s = papers["title"][papers["year"] == 2018] + papers["summary"][papers["year"] == 2018]
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.6, min_df=0.01, ngram_range=(1, 2))
vectors = vectorizer.fit_transform(acl_2018s)

words = vectorizer.get_feature_names()
acl2018_idfs = {}
for i in range(len(words)):
    if not words[i].isdigit():
        acl2018_idfs[words[i]] = vectorizer.idf_[i]

acl2018_idfs= pd.Series(acl2018_idfs)
acl2018_idfs.nlargest(10)
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score


scores = []
num_clusters = range(2, 10)
for n_clusters in num_clusters:
    labels = SpectralClustering(n_clusters=n_clusters).fit_predict(vectors)
    score = silhouette_score(vectors, labels)
    scores.append(score)

pd.Series(data=scores, index=list(num_clusters)).plot.bar()
from sklearn.decomposition import TruncatedSVD


num_cluster = 7
transformed = TruncatedSVD(n_components=2).fit_transform(vectors)
labels = SpectralClustering(n_clusters=num_cluster).fit_predict(transformed)

ax = None
colors = ["salmon", "lightskyblue", "mediumaquamarine", "wheat", "gray", "violet", "darkblue", "lime", "cadetblue"]
for cluster in range(num_cluster):
    c = pd.DataFrame(transformed[labels == cluster], columns=["pc1", "pc2"])
    if ax is None:
        ax = c.plot.scatter(x="pc1", y="pc2", color=colors[cluster], label="cluster:{}".format(cluster))
    else:
        ax = c.plot.scatter(x="pc1", y="pc2", color=colors[cluster], label="cluster:{}".format(cluster), ax=ax)