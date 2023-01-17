# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# import the used lib

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import sklearn

from matplotlib import pyplot as plt

import json

import glob

from wordcloud import WordCloud

import re

import nltk

import time

paths = os.listdir('/kaggle/input/')

print(paths)
clean_data_path = f'/kaggle/input/preprocess/clean_covid_data.csv'

df_covid = pd.read_csv(clean_data_path)
df_covid.describe(include='all')
from nltk.corpus import stopwords

import scipy.misc

from matplotlib.pyplot import imread
df_title = df_covid.loc[:, ["title"]].dropna()

df_title.info()

df_abstract = df_covid.loc[:, ["abstract"]].dropna()

df_abstract.info()
def lower_case(x):

    return x.lower()



df_title["title"] = df_title['title'].apply(lambda x: lower_case(x))

df_title["title"] = df_title['title'].apply(lambda x: x.strip())

df_title["title"] = df_title['title'].apply(lambda x: re.sub('[^a-zA-z0-9\s]','',x))

df_title["title"] = df_title['title'].apply(lambda x: re.sub(' +',' ',x))

titles = ' '.join(df_title["title"])





df_abstract["abstract"] = df_abstract['abstract'].apply(lambda x: lower_case(x))

df_abstract["abstract"] = df_abstract['abstract'].apply(lambda x: x.strip())

df_abstract["abstract"] = df_abstract['abstract'].apply(lambda x: re.sub('[^a-zA-z0-9\s]','',x))

df_abstract["abstract"] = df_abstract['abstract'].apply(lambda x: re.sub(' +',' ',x))

abstracts = ' '.join(df_abstract["abstract"])



print(titles[:100])

print(abstracts[:200])
df_title = None

df_abstract = None
stopword = stopwords.words('english')  # remove the stop words



wordcloud_title = WordCloud(max_font_size=None, background_color='white', 

                      collocations=False, stopwords=stopword,

                      width=1000, height=1000).generate(titles)



wordcloud_abstract = WordCloud(max_font_size=None, background_color='white', 

                      collocations=False, stopwords=stopword,

                      width=1000, height=1000).generate(abstracts)



plt.figure(figsize=(15,15))

plt.subplot(1,2,1)

plt.axis("off")

plt.imshow(wordcloud_title)

plt.title('Common Words in Title')

plt.subplot(1,2,2)

plt.axis("off")

plt.imshow(wordcloud_abstract)

plt.title('Common Words in Abstract')

plt.show()


wnl = nltk.stem.WordNetLemmatizer()



word_count = {}



def Pos_tag(text, publish_time):

    token = nltk.word_tokenize(text)

    pos = nltk.pos_tag(token)

    try:

        timeStruct = time.strptime(publish_time, "%Y-%m-%d")

    except:

        timeStruct = time.strptime(publish_time, "%Y")

    return token, pos, timeStruct.tm_year



# 'NN*', 'VB*'

def add2vocab(pos_tag):

    for w, p in pos_tag:

        if (re.match('NN',p) or re.match('VB',p)) and w not in stopword and w.isalnum() and len(w) > 1:

            w = w.lower()

            if re.match('NN',p):

                w = wnl.lemmatize(w, pos='n')  

            if re.match('VB',p):

                w = wnl.lemmatize(w, pos='v')

            if w in word_count:

                word_count[w] += 1

            else:

                word_count[w] = 1  
df_subset = df_covid.loc[:, ['abstract','publish_time']].dropna()

all_pos = []

all_year = []

for idx, (abstract, publish_time) in df_subset.iterrows():

    token, pos, year = Pos_tag(abstract, publish_time)

    add2vocab(pos)

    all_pos.append(pos)

    all_year.append(year)
# vocab = [k for k,v in word_count if v > 5]

word_count_sort = sorted(word_count.items(), key=lambda d: d[1], reverse=True)

vocab = [k for k,v in word_count_sort[:100]]

count = [v for k,v in word_count_sort[:100]]

print(vocab)
plt.figure(figsize=(10,10))

plt.barh(range(len(vocab[:50])), count[:50], height=0.3, color='steelblue', alpha=0.8)  

plt.yticks(range(len(vocab[:50])), vocab)

# plt.xlim(30,47)

plt.xlabel("frequency")

plt.title("Most Frequent words")

# for x, y in enumerate(count):

#     plt.text(y + 0.2, x - 0.1, '%s' % y)

plt.show()
max_year = max(all_year)

min_year = min(all_year)

print('articles are from %d year to %d year.' % (min_year, max_year))

publish_count = np.zeros(max_year-min_year+1)

for y in all_year:

    publish_count[y-min_year] += 1

year_list = list(range(min_year, max_year+1))
word2ix = {word:ix for ix, word in enumerate(vocab)}

matrix = np.zeros((max_year-min_year+1) * len(vocab)).reshape(max_year-min_year+1, len(vocab))

for pos, year in zip(all_pos, all_year):

    for w,p in pos:

        if re.match('NN',p):

            w = wnl.lemmatize(w, pos='n')

            if w in vocab:

                matrix[year-min_year][word2ix[w]] += 1

        elif re.match('VB',p):

            w = wnl.lemmatize(w, pos='v')

            if w in vocab:

                matrix[year-min_year][word2ix[w]] += 1
# sub_axix = filter(lambda x:x%200 == 0, x_axix)

small_matrix = matrix[:-1,:20].copy()  # 1957-2019, top20 words

plt.figure(figsize=(15,10))

plt.title('Words Trend')

size1, size2 = small_matrix.shape

year_num = year_list[:-1]

colors = ['g', 'r', 'b', 'k', 'y', 'c', 'm']

for idx in range(size2):

    plt.plot(year_num, list(small_matrix[:, idx]), color=colors[idx%7], label=vocab[idx])

plt.plot(year_num, publish_count[:-1], 'r*', label='publications')

plt.legend() # 显示图例



plt.xlabel('year')

plt.ylabel('word frequency')

plt.show()
small_matrix = None
small_matrix = matrix[-20:-1,:20].copy()  # 2000-2019, top20 words

small_count = publish_count[-20:-1]

for idx in range(small_matrix.shape[0]):

    small_matrix[idx,:] = small_matrix[idx,:]/small_count[idx]

plt.figure(figsize=(15,10))

plt.title('Words Trend')

size1, size2 = small_matrix.shape

year_num = year_list[-20:-1]

print(year_num)

colors = ['g', 'r', 'b', 'k', 'y', 'c', 'm']

for idx in range(size2):

    plt.plot(year_num, list(small_matrix[:, idx]), color=colors[idx%7], label=vocab[idx])

plt.legend() 



plt.xlabel('year')

plt.ylabel('word frequency')

plt.show()
matrix = None

small_matrix = None
df_cluster = df_covid.loc[:, ["paper_id","title","abstract"]].dropna()

df_cluster["abstract"] = df_cluster['abstract'].apply(lambda x: lower_case(x))

df_cluster["abstract"] = df_cluster['abstract'].apply(lambda x: x.strip())

df_cluster["abstract"] = df_cluster['abstract'].apply(lambda x: re.sub('[^a-zA-z0-9\s]','',x))

df_cluster["abstract"] = df_cluster['abstract'].apply(lambda x: re.sub(' +',' ',x))
corpus = list(df_cluster['abstract'])

print(len(corpus))
from sklearn.decomposition import TruncatedSVD

from sklearn.decomposition import SparsePCA

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction.text import HashingVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import Normalizer

from sklearn.cluster import KMeans, MiniBatchKMeans



from time import time



X = None



def prepare_text_vector(opts):

    

    # #############################################################################

    # Transform the text to vector

    

    global X

    t0 = time()

    if opts.use_hashing:

        if opts.use_idf:

            # Perform an IDF normalization on the output of HashingVectorizer

            hasher = HashingVectorizer(n_features=opts.n_features,

                                       stop_words='english', alternate_sign=False,

                                       norm=None)

            vectorizer = make_pipeline(hasher, TfidfTransformer())

        else:

            vectorizer = HashingVectorizer(n_features=opts.n_features,

                                           stop_words='english',

                                           alternate_sign=False, norm='l2')

    else:

        vectorizer = TfidfVectorizer(max_df=0.5, max_features=opts.n_features,

                                     min_df=2, stop_words='english',

                                     use_idf=opts.use_idf)

    X = vectorizer.fit_transform(corpus)

    X = X.toarray()



    print("done in %fs" % (time() - t0))

    print("n_samples: %d, n_features: %d" % X.shape)

    print()



    # #############################################################################

    # Performing dimensionality reduction

    

    if opts.n_components:

        if opts.use_pca:

            print("Performing dimensionality reduction using SparsePCA")

            t0 = time()

            pca_sk = SparsePCA(n_components=3)

            X = pca_sk.fit_transform(X)

            print("done in %fs" % (time() - t0))

        else:

            print("Performing dimensionality reduction using LSA")

            t0 = time()

            # Vectorizer results are normalized, which makes KMeans behave as

            # spherical k-means for better results. Since LSA/SVD results are

            # not normalized, we have to redo the normalization.

            svd = TruncatedSVD(opts.n_components)

            normalizer = Normalizer(copy=False)

            lsa = make_pipeline(svd, normalizer)



            X = lsa.fit_transform(X)



            print("done in %fs" % (time() - t0))



            explained_variance = svd.explained_variance_ratio_.sum()

            print("Explained variance of the SVD step: {}%".format(

                int(explained_variance * 100)))

    print()

    
from mpl_toolkits.mplot3d import Axes3D

from sklearn.cluster import KMeans



def cluster_text(opts):  

    

    # #############################################################################

    # Kmeans clustering



    # y_preds = KMeans(n_clusters=5, random_state=0, n_jobs=4, verbose=10).fit_predict(pca_result)

    if opts.minibatch:

        estimators = [('k_means_3', MiniBatchKMeans(n_clusters=3, init='k-means++', n_init=1,

                                                    init_size=1000, batch_size=1000, verbose=opts.verbose), '221'),

                      ('k_means_4', MiniBatchKMeans(n_clusters=4, init='k-means++', n_init=1,

                                                    init_size=1000, batch_size=1000, verbose=opts.verbose), '222'),

                      ('k_means_5', MiniBatchKMeans(n_clusters=5, init='k-means++', n_init=1,

                                                    init_size=1000, batch_size=1000, verbose=opts.verbose), '223'),

                      ('k_means_6', MiniBatchKMeans(n_clusters=6, init='k-means++', n_init=1,

                                                    init_size=1000, batch_size=1000, verbose=opts.verbose), '224'),]

    else:

        estimators = [('k_means_3', KMeans(n_clusters=3, init='k-means++', max_iter=100, 

                                           n_init=1, verbose=opts.verbose), '221'),

                      ('k_means_4', KMeans(n_clusters=4, init='k-means++', max_iter=100, 

                                           n_init=1, verbose=opts.verbose), '222'),

                      ('k_means_5', KMeans(n_clusters=5, init='k-means++', max_iter=100, 

                                           n_init=1, verbose=opts.verbose), '223'),

                      ('k_means_6', KMeans(n_clusters=6, init='k-means++', max_iter=100, 

                                           n_init=1, verbose=opts.verbose), '224'),]

    fignum = 1

    titles = ['Covid-19 Articles\' Abstract - Clustered (K-Means) 3 clusters', 

              'Covid-19 Articles\' Abstract - Clustered (K-Means) 4 clusters', 

              'Covid-19 Articles\' Abstract - Clustered (K-Means) 5 clusters', 

              'Covid-19 Articles\' Abstract - Clustered (K-Means) 6 clusters']

    fig = plt.figure(fignum, figsize=(20, 15))

    t0 = time()

    for name, est, fig_idx in estimators:

        ax = fig.add_subplot(int(fig_idx),projection='3d')

        print("Clustering sparse data with %s" % est)

        est.fit(X)

        labels = est.labels_

        centers = est.cluster_centers_

        ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=10, c=labels.astype(np.float), edgecolor='y')

        ax.w_xaxis.set_ticklabels([])

        ax.w_yaxis.set_ticklabels([])

        ax.w_zaxis.set_ticklabels([])

        ax.set_xlabel('x')

        ax.set_ylabel('y')

        ax.set_zlabel('z')

        ax.set_title(titles[fignum - 1])

        ax.dist = 12

        fignum = fignum + 1

    print("done in %0.3fs" % (time() - t0))

    fig.show()
from optparse import OptionParser



opt = OptionParser()

opt.add_option("--use_hashing",

              action="store_true", default=False,

              help="Use a hashing feature vectorizer")

opt.add_option("--n_features", type=int, default=10000,

              help="Maximum number of features (dimensions)"

                   " to extract from text.")

opt.add_option("--use_idf",

              action="store_false", default=True,

              help="Use Inverse Document Frequency feature weighting.")



opt.add_option("--n_components", type="int", default=3,

               help="Preprocess documents with latent semantic analysis.")

opt.add_option("--use_pca", default=True, 

               help="if True use PCA else use SVD")

opt.add_option("--minibatch", action="store_false", default=False,

              help="Use ordinary k-means algorithm (in batch mode).")



opt.add_option("--verbose",

              action="store_true", dest="verbose", default=False,

              help="Print progress reports inside k-means algorithm.")

opt.use_hashing = True

opt.n_features = 2**10

opt.use_idf = False

opt.n_components = 3

opt.use_pca = True

opt.minibatch = False

opt.verbose = False

prepare_text_vector(opt)

print(X.shape)

cluster_text(opt)
opt.use_hashing = True

opt.n_features = 2**12

opt.use_idf = False

opt.n_components = 3

opt.use_pca = True

opt.minibatch = False

opt.verbose = False

prepare_text_vector(opt)

print(X.shape)

cluster_text(opt)
opt.use_hashing = True

opt.n_features = 2**10

opt.use_idf = True

opt.n_components = 3

opt.use_pca = True

opt.minibatch = False

opt.verbose = False

prepare_text_vector(opt)

print(X.shape)

cluster_text(opt)
opt.use_hashing = True

opt.n_features = 2**10

opt.use_idf = False

opt.n_components = 3

opt.use_pca = False

opt.minibatch = False

opt.verbose = False

prepare_text_vector(opt)

print(X.shape)

cluster_text(opt)
opt.use_hashing = True

opt.n_features = 2**10

opt.use_idf = False

opt.n_components = 3

opt.use_pca = False

opt.minibatch = False

opt.verbose = False

prepare_text_vector(opt)

print(X.shape)
est = KMeans(n_clusters=3, init='k-means++', max_iter=100, n_init=1, verbose=opt.verbose)

est.fit(X)

labels = est.labels_

centers = est.cluster_centers_

center_idx = []

for center in centers:

    min_idx = 0

    min_dis = np.inf

    for idx, loc in enumerate(X):

        dis = ((center - loc)**2).sum()

        if dis < min_dis:

            min_dis = dis

            min_idx = idx

    center_idx.append(min_idx)

print(center_idx)



pd.set_option('display.width',400)

for c_ix in center_idx:

    tmp = df_cluster.iloc[c_ix:c_ix+1]

    title = list(tmp['title'])

    print(' '.join(title).replace('<br>', ' '))

#     abstract = list(tmp['abstract'])

#     print(' '.join(abstract))

cluster0_idx = [idx for idx, x in enumerate(labels) if x == 0]

cluster1_idx = [idx for idx, x in enumerate(labels) if x == 1]

cluster2_idx = [idx for idx, x in enumerate(labels) if x == 2]

print(len(cluster0_idx))

print(len(cluster1_idx))

print(len(cluster2_idx))
cluster0_abstract = list(df_cluster.iloc[cluster0_idx].loc[:, 'abstract'])

cluster1_abstract = list(df_cluster.iloc[cluster1_idx].loc[:, 'abstract'])

cluster2_abstract = list(df_cluster.iloc[cluster2_idx].loc[:, 'abstract'])

assert len(cluster0_abstract) == len(cluster0_idx) and len(cluster1_abstract) == len(cluster1_idx) and len(cluster2_abstract) == len(cluster2_idx)

abstract0 = ' '.join(cluster0_abstract)

abstract1 = ' '.join(cluster1_abstract)

abstract2 = ' '.join(cluster2_abstract)
wordcloud_abstract0 = WordCloud(max_font_size=None, background_color='white', 

                      collocations=False, stopwords=stopword,

                      width=1000, height=1000).generate(abstract0)

wordcloud_abstract1 = WordCloud(max_font_size=None, background_color='white', 

                      collocations=False, stopwords=stopword,

                      width=1000, height=1000).generate(abstract1)

wordcloud_abstract2 = WordCloud(max_font_size=None, background_color='white', 

                      collocations=False, stopwords=stopword,

                      width=1000, height=1000).generate(abstract2)



plt.figure(figsize=(25,15))

plt.subplot(1,3,1)

plt.axis("off")

plt.imshow(wordcloud_abstract0)

plt.title('Common Words in Abstract Cluster 1')

plt.subplot(1,3,2)

plt.axis("off")

plt.imshow(wordcloud_abstract1)

plt.title('Common Words in Abstract Cluster 2')

plt.subplot(1,3,3)

plt.axis("off")

plt.imshow(wordcloud_abstract2)

plt.title('Common Words in Abstract Cluster 3')

plt.show()
w0 = [w for w,f in list(wordcloud_abstract0.words_.items())[:10]]

print(', '.join(w0))

w1 = [w for w,f in list(wordcloud_abstract1.words_.items())[:10]]

print(', '.join(w1))

w2 = [w for w,f in list(wordcloud_abstract2.words_.items())[:10]]

print(', '.join(w2))