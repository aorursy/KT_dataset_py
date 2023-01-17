import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns

import sklearn
import re
import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords

import os
print(os.listdir("../input"))
stopwords = stopwords.words('english')
stemmer = SnowballStemmer("english")
# Utility functions

# takes raw data, converts to sentences, and removes tokens that aren't words or digits
def clean_sentences(text):
    tokens = [sent for sent in nltk.sent_tokenize(text)]

    sent_list = []
    for sent in tokens:
        sent_str = ''
        for i, word in enumerate(nltk.word_tokenize(sent)):
            # nltk doesn't handle apostrophes correctly
            if word[0] == "'":
                sent_str = sent_str[:-1]
            
            # only adds words and digits
            if re.search('[a-zA-Z0-9]', word):
                sent_str += word.lower() + ' '
        sent_list.append(sent_str.strip())

    return sent_list

# takes list of clean sentences and converts to list of tokens
def tokens_only(text):
    tokens = []
    
    for sentence in text:
        tokens.extend(sentence.split(" "))
    
    return tokens

# takes in text, cleans it, and returns lemma only
def lemma_tokens(text):
    tokens = tokens_only(clean_sentences(text))
    
    return [stemmer.stem(token) for token in tokens]
# loading files
filenames = ['against_the_gods', 'battle_through_the_heavens', 'desolate_era', 'emperors_domination', 'martial_god_asura', 'martial_world', 'overgeared', 'praise_the_orc', 'sovereign_of_the_three_realms', 'wu_dong_qian_kun']
raw_files = []

for filename in filenames:
    with open('../input/' + filename + '.txt', encoding='utf-8') as myfile:
        raw_files.append(myfile.read())
# use extend so it's a big flat list of vocab
all_lemma = []
all_tokens = []
all_sentences = []
all_sentences_label = []

for i, doc in enumerate(raw_files):
    
    # clean sentences    
    tmp_list= clean_sentences(doc)
    all_sentences.extend(tmp_list)
    for j in range(len(tmp_list)):
        all_sentences_label.append(filenames[i])
    
    # convert list of clean sentences to tokens
    tmp_list = tokens_only(tmp_list)
    all_tokens.extend(tmp_list)
    
    # gets root word for tokens in document
    all_lemma.extend(lemma_tokens(doc))
vocab_df = pd.DataFrame({'words': all_tokens}, index = all_lemma)
print ('there are', vocab_df.shape[0], 'words and', len(all_sentences), 'sentences')
vocab_df.head()
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_df=0.75, # drop words that occur in more than 3/4 of the sentence
                             min_df=2, # only use words that appear at least twice
                             stop_words='english', 
                             lowercase=False,
                             use_idf=True, # use inverse document frequencies in our weighting
                             norm=u'l2', # Apply a correction factor so that longer sentences and shorter sentences get treated equally
                             smooth_idf=True, # Adds 1 to all document frequencies, as if an extra document existed that used every word once.  Prevents divide-by-zero errors
                             tokenizer=lemma_tokens,
                             ngram_range=(1,3)                             
                            )

# Apply the vectorizer
tfidf_matrix = vectorizer.fit_transform(raw_files)
print(tfidf_matrix.shape)
terms = vectorizer.get_feature_names()
from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth, SpectralClustering, AffinityPropagation

num_clusters = 5
km = KMeans(n_clusters=num_clusters)
km.fit(tfidf_matrix)

clusters = km.labels_.tolist()
novels = { 'title': filenames, 'text': raw_files, 'cluster': clusters }
df = pd.DataFrame(novels, index = [clusters] , columns = ['title', 'text', 'cluster'])
print("Clusters:")
print('-'*40)

#sort cluster centers by proximity to centroid
order_centroids = km.cluster_centers_.argsort()[:, ::-1] 

for i in range(num_clusters):
    if i != 0:
        print('\n')
    print("Cluster %d words: " % i, end='')
    
    for j, ind in enumerate(order_centroids[i, :10]):
        if (j == 0):
            print('%s' % vocab_df.loc[terms[ind].split(' ')].values.tolist()[0][0], end='')
        else:
            print(', %s' % vocab_df.loc[terms[ind].split(' ')].values.tolist()[0][0], end='')
    print()
    
    print("Cluster %d titles: " % i, end='')
    for j, title in enumerate(df.loc[i]['title'].values.tolist()):
        if (j == 0):
            print('%s' % title, end='')
        else:
            print(', %s' % title, end='')
ms = MeanShift()
ms.fit(tfidf_matrix.toarray())

n_clusters_ = len(np.unique(ms.labels_))
print("Number of estimated clusters: {}".format(n_clusters_))
sc = SpectralClustering(n_clusters=num_clusters)
sc.fit(tfidf_matrix)
clusters = sc.labels_.tolist()
sc_df = pd.DataFrame(sc.affinity_matrix_, index=filenames, columns=filenames)
print(sns.heatmap(sc_df.corr()))
sc_df
novels = { 'title': filenames, 'text': raw_files, 'cluster': clusters }

df = pd.DataFrame(novels, index = [clusters] , columns = ['title', 'text', 'cluster'])
print("Clusters:")
print('-'*40)

for i in range(num_clusters):
    if i != 0:
        print('\n')

    print("Cluster %d titles: " % i, end='')
    for j, title in enumerate(df.loc[i]['title'].values.tolist()):
        if (j == 0):
            print('%s' % title, end='')
        else:
            print(', %s' % title, end='')
num_clusters = 6
sc = SpectralClustering(n_clusters=num_clusters)
sc.fit(tfidf_matrix)
clusters = sc.labels_.tolist()
novels = { 'title': filenames, 'text': raw_files, 'cluster': clusters }

df = pd.DataFrame(novels, index = [clusters] , columns = ['title', 'text', 'cluster'])
print("Clusters:")
print('-'*40)

for i in range(num_clusters):
    if i != 0:
        print('\n')

    print("Cluster %d titles: " % i, end='')
    for j, title in enumerate(df.loc[i]['title'].values.tolist()):
        if (j == 0):
            print('%s' % title, end='')
        else:
            print(', %s' % title, end='')
af = AffinityPropagation()
af.fit(tfidf_matrix)

af_clusters = len(af.cluster_centers_indices_)
print("Number of estimated clusters: {}".format(af_clusters))
clusters = af.labels_
novels = { 'title': filenames, 'text': raw_files, 'cluster': clusters }

df = pd.DataFrame(novels, index = [clusters] , columns = ['title', 'text', 'cluster'])
print("Clusters:")
print('-'*40)

for i in range(af_clusters):
    if i != 0:
        print('\n')

    print("Cluster %d titles: " % i, end='')
    for j, title in enumerate(df.loc[i]['title'].values.tolist()):
        if (j == 0):
            print('%s' % title, end='')
        else:
            print(', %s' % title, end='')
num_clusters = 4
km = KMeans(n_clusters=num_clusters)
km.fit(tfidf_matrix)

clusters = km.labels_.tolist()
novels = { 'title': filenames, 'text': raw_files, 'cluster': clusters }

df = pd.DataFrame(novels, index = [clusters] , columns = ['title', 'text', 'cluster'])
print("Clusters:")
print('-'*40)

#sort cluster centers by proximity to centroid
order_centroids = km.cluster_centers_.argsort()[:, ::-1] 

for i in range(num_clusters):
    if i != 0:
        print('\n')
    print("Cluster %d words: " % i, end='')
    
    for j, ind in enumerate(order_centroids[i, :10]):
        if (j == 0):
            print('%s' % vocab_df.loc[terms[ind].split(' ')].values.tolist()[0][0], end='')
        else:
            print(', %s' % vocab_df.loc[terms[ind].split(' ')].values.tolist()[0][0], end='')
    print()
    
    print("Cluster %d titles: " % i, end='')
    for j, title in enumerate(df.loc[i]['title'].values.tolist()):
        if (j == 0):
            print('%s' % title, end='')
        else:
            print(', %s' % title, end='')
vectorizer = TfidfVectorizer(max_df=0.75, # drop words that occur in more than 3/4 of the sentence
                             min_df=2, # only use words that appear at least twice
                             stop_words='english', 
                             lowercase=False,
                             use_idf=True, # use inverse document frequencies in our weighting
                             norm=u'l2', # Apply a correction factor so that longer sentences and shorter sentences get treated equally
                             smooth_idf=True, # Adds 1 to all document frequencies, as if an extra document existed that used every word once.  Prevents divide-by-zero errors
                             tokenizer=lemma_tokens                        
                            )

# Apply the vectorizer
tfidf_matrix = vectorizer.fit_transform(all_sentences)
print("Number of features: %d" % tfidf_matrix.get_shape()[1])
terms = vectorizer.get_feature_names()
# splitting into training and test sets
# keeps sentence structure
X_train, X_test, y_train, y_test = train_test_split(all_sentences, all_sentences_label, test_size=0.25, random_state=0)
# scores of tfidf
X_train_tfidf, X_test_tfidf, y_train_tfidf, y_test_tfidf = train_test_split(tfidf_matrix, all_sentences_label, test_size=0.25, random_state=0)

# force output into compressed sparse row if it isn't already; readable format
X_train_tfidf_csr = X_train_tfidf.tocsr()
n = X_train_tfidf_csr.shape[0]
terms = vectorizer.get_feature_names()

# create empty list of dictionary, per sentence
sents_tfidf = [{} for _ in range(0,n)]

# for each sentence, list feature words and tf-idf score
for i, j in zip(*X_train_tfidf_csr.nonzero()):
    sents_tfidf[i][terms[j]] = X_train_tfidf_csr[i, j]
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

# SVD data reducer.  Reduces the number of features from 5252 to 950.
svd= TruncatedSVD(950)
lsa = make_pipeline(svd, Normalizer(copy=False))
# Run SVD on the training data, then project the training data.
lsa_all_sents = lsa.fit_transform(tfidf_matrix)

variance_explained=svd.explained_variance_ratio_
total_variance = variance_explained.sum()
print("Percent variance captured by all components:",total_variance*100)

#Looking at what sorts of sentences solution considers similar, for the first three identified topics
sents_by_component=pd.DataFrame(lsa_all_sents,index=all_sentences)
for i in range(3):
    print('Component {}:'.format(i))
    print(sents_by_component.loc[:,i].sort_values(ascending=False)[0:10])
X_train_svd, X_test_svd, y_train_svd, y_test_svd = train_test_split(sents_by_component, all_sentences_label, test_size=0.25, random_state=0)
from sklearn import ensemble
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
# logistic regression with tf idf
lr = LogisticRegression()

lr.fit(X_train_tfidf, y_train_tfidf)
print('Training set score:', lr.score(X_train_tfidf, y_train_tfidf))
print('Test set score:', lr.score(X_test_tfidf, y_test_tfidf))
# logistic regression with SVD
lr.fit(X_train_svd, y_train_svd)
print('Training set score:', lr.score(X_train_svd, y_train_svd))
print('Test set score:', lr.score(X_test_svd, y_test_svd))
# random forest with tf idf
rfc = ensemble.RandomForestClassifier()

rfc.fit(X_train_tfidf, y_train_tfidf)
print('Training set score:', rfc.score(X_train_tfidf, y_train_tfidf))
print('Test set score:', rfc.score(X_test_tfidf, y_test_tfidf))
# random forest with SVD
rfc.fit(X_train_svd, y_train_svd)
print('Training set score:', rfc.score(X_train_svd, y_train_svd))
print('Test set score:', rfc.score(X_test_svd, y_test_svd))
# support vector classification with tf idf
svc = SVC(kernel='linear', gamma='auto')

svc.fit(X_train_tfidf, y_train_tfidf)
print('Training set score:', svc.score(X_train_tfidf, y_train_tfidf))
print('Test set score:', svc.score(X_test_tfidf, y_test_tfidf))
# support vector classification with svd
svc.fit(X_train_svd, y_train_svd)
print('Training set score:', svc.score(X_train_svd, y_train_svd))
print('Test set score:', svc.score(X_test_svd, y_test_svd))
test_list = []
for title in filenames:
    tmp_list = ""
    for i, sentence_label in enumerate(y_test):
        if sentence_label == title:
            tmp_list += X_test[i] + '. '
    test_list.append(tmp_list)        
vectorizer = TfidfVectorizer(max_df=0.75, # drop words that occur in more than 3/4 of the sentence
                             min_df=2, # only use words that appear at least twice
                             stop_words='english', 
                             lowercase=False,
                             use_idf=True, # use inverse document frequencies in our weighting
                             norm=u'l2', # Apply a correction factor so that longer sentences and shorter sentences get treated equally
                             smooth_idf=True, # Adds 1 to all document frequencies, as if an extra document existed that used every word once.  Prevents divide-by-zero errors
                             tokenizer=lemma_tokens,
                             ngram_range=(1,3)                             
                            )

# Apply the vectorizer
tfidf_matrix = vectorizer.fit_transform(test_list)
print(tfidf_matrix.shape)
terms = vectorizer.get_feature_names()
km = KMeans(n_clusters=num_clusters)
km.fit(tfidf_matrix)

clusters = km.labels_.tolist()
novels = { 'title': filenames, 'text': test_list, 'cluster': clusters }
df = pd.DataFrame(novels, index = [clusters] , columns = ['title', 'text', 'cluster'])
print("Clusters:")
print('-'*40)

#sort cluster centers by proximity to centroid
order_centroids = km.cluster_centers_.argsort()[:, ::-1] 

for i in range(num_clusters):
    if i != 0:
        print('\n')
    print("Cluster %d words: " % i, end='')
    
    for j, ind in enumerate(order_centroids[i, :10]):
        if (j == 0):
            print('%s' % vocab_df.loc[terms[ind].split(' ')].values.tolist()[0][0], end='')
        else:
            print(', %s' % vocab_df.loc[terms[ind].split(' ')].values.tolist()[0][0], end='')
    print()
    
    print("Cluster %d titles: " % i, end='')
    for j, title in enumerate(df.loc[i]['title'].values.tolist()):
        if (j == 0):
            print('%s' % title, end='')
        else:
            print(', %s' % title, end='')