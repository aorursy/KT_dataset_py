# load all libraries used in later analysis
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['figure.figsize'] = (10, 6)
from matplotlib import style
style.use('ggplot')
import spacy
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN, MeanShift, MiniBatchKMeans
import os
from pprint import pprint
import string
import re
from sklearn.decomposition import PCA
from collections import Counter
# load data
loans = pd.read_csv("../input/kiva_loans.csv")
loans.head().transpose()
# aggregate "use" by country and region and combine use text
use_by_CR = loans[['country', 'region', 'use']] \
    .replace(np.nan, "") \
    .groupby(['country', 'region'])['use'] \
    .apply(lambda x: "\n".join(x)) \
    .reset_index()  # normalise
use_by_CR['region'].replace("", "#other#", inplace=True)
use_by_CR['country'].replace("", "#other#", inplace=True)
# generate a combined field for aggregation purposes
use_by_CR['CR'] = use_by_CR['country'] + "_" + use_by_CR['region']
# now we use spacy to process the per-region use descriptions and obtain document vectors
nlp = spacy.load('en_core_web_lg', disable=["tagger", "parser", "ner"])

raw_use_texts = list(use_by_CR['use'].values)
processed_use_texts = [nlp(text) for text in raw_use_texts]
processed_use_vectors = np.array([text.vector for text in processed_use_texts])
processed_use_vectors.shape
# we would like to map the document vectors into a space where we can compare their
# similarities and visualise them. A 2-D space is preferable. therefore we perform a TSNE
# transformation to flatten the 300-D vector space.
tsne = TSNE(n_components=2, metric='cosine', random_state=7777)
fitted = tsne.fit(processed_use_vectors)
fitted_components = fitted.embedding_
fitted_components.shape
use_by_CR['cx'] = fitted_components[:, 0]
use_by_CR['cy'] = fitted_components[:, 1]
use_by_CR.head()
# now we plot all the transformed points for each country against each other, for countries
# with the most recorded regions
country_region_cnt = use_by_CR.groupby('country').size()
selected_countries = country_region_cnt[country_region_cnt > 150]
n_selected_countries = len(selected_countries)
selected_country_pos = np.where(country_region_cnt > 150)[0]
id2country = dict(enumerate(selected_countries.index))
country2id = {v: k for k, v in id2country.items()}
selected_use_by_CR = use_by_CR.query('country in @selected_countries.index')
fig, ax = plt.subplots(figsize=(16, 12))
plt.scatter(selected_use_by_CR['cx'], selected_use_by_CR['cy'], s=15,
            c=[country2id[x] for x in selected_use_by_CR['country']],
            cmap=plt.cm.get_cmap('tab20', 19))
formatter = plt.FuncFormatter(lambda val, loc: id2country[val])
plt.colorbar(ticks=np.arange(19), format=formatter);
plt.show()
# select all aggregated use cases in Mali:
mali_regional_uses = use_by_CR.query('country == "Mali"')
mali_regional_uses.shape
cluster = KMeans(n_clusters=4, random_state=7777)
cluster.fit_transform(mali_regional_uses[['cx', 'cy']]);
fig, ax = plt.subplots(figsize=(16, 12))
plt.scatter(mali_regional_uses['cx'], mali_regional_uses['cy'], s=15,
            c=cluster.labels_,
            cmap=plt.cm.get_cmap('tab10', 4))
formatter = plt.FuncFormatter(lambda val, loc: "Cluser {0}".format(val))
plt.colorbar(ticks=np.arange(4), format=formatter);
plt.show()
# let use see if the clusters are indeed different
# examples from cluster 0
for region_uses in mali_regional_uses['use'].iloc[cluster.labels_ == 0].iloc[:10]:
    print(region_uses[:min(500, len(region_uses))], end="")
    print('...' if len(region_uses) > 500 else "")
    print('-' * 20)
# examples from cluster 1
for region_uses in mali_regional_uses['use'].iloc[cluster.labels_ == 1].iloc[:10]:
    print(region_uses[:min(500, len(region_uses))], end="")
    print('...' if len(region_uses) > 500 else "")
    print('-' * 20)
# examples from cluster 2
for region_uses in mali_regional_uses['use'].iloc[cluster.labels_ == 2].iloc[:10]:
    print(region_uses[:min(500, len(region_uses))], end="")
    print('...' if len(region_uses) > 500 else "")
    print('-' * 20)
# examples from cluster 3
for region_uses in mali_regional_uses['use'].iloc[cluster.labels_ == 3].iloc[:10]:
    print(region_uses[:min(500, len(region_uses))], end="")
    print('...' if len(region_uses) > 500 else "")
    print('-' * 20)
guatemala_regional_uses = use_by_CR.query('country == "Guatemala"')
guatemala_regional_uses.shape
cluster2 = KMeans(n_clusters=5, random_state=7777)
cluster2.fit_transform(guatemala_regional_uses[['cx', 'cy']]);
fig, ax = plt.subplots(figsize=(16, 12))
plt.scatter(guatemala_regional_uses['cx'], guatemala_regional_uses['cy'], s=15,
            c=cluster2.labels_,
            cmap=plt.cm.get_cmap('tab10', 5))
formatter = plt.FuncFormatter(lambda val, loc: "Cluser {0}".format(val))
plt.colorbar(ticks=np.arange(5), format=formatter);
plt.show()
for region_uses in guatemala_regional_uses['use'].iloc[cluster2.labels_ == 1].iloc[:10]:
    cleaned = re.sub(r'(\n\s*)+\n+', '\n', region_uses)  # remove excessive empty lines 
    # caused by missing data
    print(cleaned[:min(500, len(cleaned))], end="")
    print('...' if len(cleaned) > 500 else "")
    print('-' * 20)
combined_data = pd.concat([guatemala_regional_uses.iloc[cluster2.labels_ == 1],
                          mali_regional_uses.iloc[cluster.labels_ == 1]], axis=0)
i2c = {0: "Mali", 1: "Guatemala"}
c2i = {v: k for k, v in i2c.items()}
fig, ax = plt.subplots(figsize=(16, 12))
plt.scatter(combined_data['cx'], combined_data['cy'], s=15,
            c=[c2i[x] for x in combined_data['country']],
            cmap=plt.cm.get_cmap('tab10', 2))
formatter = plt.FuncFormatter(lambda val, loc: i2c[val])
plt.colorbar(ticks=np.arange(2), format=formatter);
plt.show()
def count_resales(s):
    resales_words = ["resell", "sell", "resale"]
    return len(re.findall("|".join(["(?:{0})".format(x) for x in resales_words]), s))
print("Mentions of resales in Mali:")
print(mali_regional_uses.iloc[cluster.labels_ == 1]["use"].apply(count_resales).sum())
print("Mentions of resales in Guatemala:")
print(guatemala_regional_uses.iloc[cluster2.labels_ == 1]["use"].apply(count_resales).sum())
samoa = use_by_CR.query('country == "Samoa"')
fig, ax = plt.subplots(figsize=(16, 12))
plt.scatter(samoa['cx'], samoa['cy'], s=15)
plt.show()
for region_uses in samoa['use'][:10]:
    cleaned = re.sub(r'(\n\s*)+\n+', '\n', region_uses)
    print(cleaned[:min(500, len(cleaned))], end="")
    print('...' if len(cleaned) > 500 else "")
    print('-' * 20)
kyrg = use_by_CR.query('country == "Kyrgyzstan"')
for region_uses in kyrg['use'][:10]:
    cleaned = re.sub(r'(\n\s*)+\n+', '\n', region_uses)
    print(cleaned[:min(500, len(cleaned))], end="")
    print('...' if len(cleaned) > 500 else "")
    print('-' * 20)
def count_livestock(s):
    resales_words = ["sheep", "cow", "calf", "calves", "bull", "livestock"]
    return len(re.findall("|".join(["(?:{0})".format(x) for x in resales_words]), s))
livestock_counts = use_by_CR.copy()
livestock_counts['n_words'] = livestock_counts['use'] \
    .apply(lambda x: len(re.findall(r"\w+", x)))
livestock_counts['n_livestock'] = livestock_counts["use"].apply(count_livestock)
livestock_totals = livestock_counts.groupby('country')[['n_words', 'n_livestock']].sum()
livestock_totals['ratio'] = livestock_totals['n_livestock'] / livestock_totals['n_words']
livestock_totals.sort_values("ratio", ascending=False)[:10]
print("document vectors dimensions: {0}".format(processed_use_vectors.shape))
pca = PCA(n_components=100, random_state=7777)
pca.fit(processed_use_vectors)
N = 20
sns.barplot(x=pca.explained_variance_ratio_[:N], y=["C_{0}".format(x) for x in range(N)])
N = 35
print("number of selected PCs: {}".format(N))
print("total % variance explained: {}".format(np.sum(pca.explained_variance_ratio_[:N])))
low_dim_vecs = pca.transform(processed_use_vectors)[:, :N]
print(low_dim_vecs.shape)
ldcls = MiniBatchKMeans(n_clusters=50, random_state=7777)
ldcls.fit(low_dim_vecs);
use_by_CR['cluster'] = ldcls.labels_
print(Counter(ldcls.labels_))
N = len(np.unique(ldcls.labels_))
fig, ax = plt.subplots(figsize=(16, 12))
plt.scatter(use_by_CR['cx'], use_by_CR['cy'], s=15,
            c=ldcls.labels_ + 1,
            cmap=plt.cm.get_cmap('rainbow', N))
formatter = plt.FuncFormatter(lambda val, loc: "C_{}".format(val))
plt.colorbar(ticks=np.arange(N), format=formatter);
plt.show()
