################################################# import libraries ###########################################

import pandas as pd
import os
from nltk.corpus import stopwords
import string
import re
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import adjusted_rand_score
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import plotly.plotly as py
import operator
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
import time
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.cluster import AgglomerativeClustering
def rem_sw(df):
    # Downloading stop words
    stop_words = set(stopwords.words('english'))

    # Removing Stop words from training data
    count = 0
    for sentence in df:
        sentence = [word for word in sentence.lower().split() if word not in stop_words]
        sentence = ' '.join(sentence)
        df.loc[count] = sentence
        count+=1
    return(df)
def rem_punc(df):
    count = 0
    for s in df:
        cleanr = re.compile('<.*?>')
        s = re.sub(r'\d+', '', s)
        s = re.sub(cleanr, '', s)
        s = re.sub("'", '', s)
        s = re.sub(r'\W+', ' ', s)
        s = s.replace('_', '')
        df.loc[count] = s
        count+=1
    return(df)
def lemma(df):

    lmtzr = WordNetLemmatizer()

    count = 0
    stemmed = []
    for sentence in df:    
        word_tokens = word_tokenize(sentence)
        for word in word_tokens:
            stemmed.append(lmtzr.lemmatize(word))
        sentence = ' '.join(stemmed)
        df.iloc[count] = sentence
        count+=1
        stemmed = []
    return(df)
def stemma(df):

    stemmer = SnowballStemmer("english") #SnowballStemmer("english", ignore_stopwords=True)

    count = 0
    stemmed = []
    for sentence in df:
        word_tokens = word_tokenize(sentence)
        for word in word_tokens:
            stemmed.append(stemmer.stem(word))
        sentence = ' '.join(stemmed)
        df.iloc[count] = sentence
        count+=1
        stemmed = []
    return(df)
df_master = pd.read_csv("../input/imdb_master.csv", encoding='latin-1', index_col = 0)
df_master = df_master[df_master.label != 'unsup']
imdb_train = df_master[df_master['type'] == 'train'].copy()
imdb_test =  df_master[df_master['type'] == 'test'].copy()
imdb_train['review'] = rem_sw(imdb_train['review'])
imdb_test['review'] = rem_sw(imdb_test['review'])

imdb_train['review'] = rem_punc(imdb_train['review'])
imdb_test['review'] = rem_punc(imdb_test['review'])

imdb_train['review'] = lemma(imdb_train['review'])
imdb_train['review'] = stemma(imdb_train['review'])

imdb_test['review'] = lemma(imdb_test['review'])
imdb_test['review'] = stemma(imdb_test['review'])
from gensim.models import Word2Vec

model = Word2Vec(imdb_train['review'].apply(lambda s: s.split()))
model.save("word2vec.model")
model.wv.most_similar('movi')
model.wv.words_closer_than('actor', 'star')
model.wv.similarity('actor', 'star')
kmeans_args = {
    'n_clusters': 1000,
}

clustering = KMeans(**kmeans_args).fit_predict(model.wv.vectors)
word2centroid = {k: v for k, v in zip(model.wv.index2word, clustering)}
from numpy import zeros

def make_bag_of_centroids(sentence, word_centroid_map, cluster_size):
    centroids = zeros(cluster_size, dtype="float32")

    for word in sentence:
        if word in word_centroid_map:
            centroids[word_centroid_map[word]] += 1

    return centroids

as_centroid = lambda s: make_bag_of_centroids(s.split(), word2centroid, kmeans_args['n_clusters'])
imdb_train[:1].review.apply(as_centroid).tolist()
from xgboost import XGBClassifier
from sklearn.preprocessing import scale

fit = XGBClassifier().fit(scale(imdb_train.review.apply(as_centroid).tolist()), imdb_train.label)
predictions = fit.predict(scale(imdb_test.review.apply(as_centroid).tolist()))
sum(True for a,b in zip(predictions, imdb_test.label) if a == b) / len(imdb_test), sum(True for a,b in zip(predictions, imdb_test.label) if a != b) / len(imdb_test)
imdb_test['prediction'] = predictions
# from http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py

import itertools
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
cnf_matrix = confusion_matrix(imdb_test.label, predictions)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['pos', 'neg'],
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['pos', 'neg'], normalize=True,
                      title='Normalized confusion matrix')
imdb_test[imdb_test.label != imdb_test.prediction]