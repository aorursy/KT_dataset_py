# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from tqdm import tqdm_notebook as tqdm

import seaborn as sns

from matplotlib import pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

print(os.listdir("../input/jigsaw-toxic-comment-classification-challenge"))



# Any results you write to the current directory are saved as output.
# Load the dataset

train = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/train.csv.zip')

target = train['toxic']

test = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/test.csv.zip')



train.head(10)
sns.distplot(target, kde=False)

print(target.mean())

print('Minimum accuracy:', max(target.mean(), 1 - target.mean()))
# define preprocessing function

import string



def preprocess(doc):

    # lowercasing

    doc = doc.lower()

    # remove punctuation and different kinds of whitespaces e.g. newlines and tabs

    for p in string.punctuation + string.whitespace:

        doc = doc.replace(p, ' ')

    # remove unneeded spaces

    doc = doc.strip()

    doc = ' '.join([w for w in doc.split(' ') if w != ''])

    return doc
# Preprocessed text corpus

corpus = train['comment_text'].map(preprocess)

corpus.head(10)
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.decomposition import TruncatedSVD



vectorizer = TfidfVectorizer(max_features=30000)

svd = TruncatedSVD(n_components=100)



X_tfidf = vectorizer.fit_transform(corpus)

print(X_tfidf.shape)

X_svd = svd.fit_transform(X_tfidf)

print(X_svd.shape)
from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier



from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from sklearn.model_selection import cross_validate

from sklearn.utils.testing import ignore_warnings



def eval_on_trainset(X, y, model_names=None):

    models = {

        'SVM_rbf': SVC(C=100, kernel='rbf'),

        'SVM_linear': SVC(C=100, kernel='linear'),

        'Log regression': LogisticRegression(),

        'naive bayes': GaussianNB(),

        'random forest': RandomForestClassifier(),

        'KNN': KNeighborsClassifier(),

    }

    scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

    

    print('Dummy accuracy:', max(y.mean(), 1 - y.mean()))

    print()

    

    try:

        X = X.toarray()

    except:

        pass

    

    for name in model_names or sorted(models):

        model = models[name]

        with ignore_warnings():

            scores_svc = cross_validate(model, X, y, cv=3, scoring=scoring)

        for sc in scoring:

            mean = scores_svc['test_' + sc].mean()

            std = scores_svc['test_' + sc].std()

            print(name, sc, '{:.03} +- {:.03}'.format(mean, std))

        print()
# Small part of tfidf, because of time and memory usage

%time eval_on_trainset(X_tfidf[:3000], target[:3000], ['Log regression', 'naive bayes', 'random forest'])
# Part of data, because of time usage

%time eval_on_trainset(X_svd[:10000], target[:10000])
# Eval on full data, for fastest algorythms

%time eval_on_trainset(X_svd, target, ['Log regression', 'naive bayes', 'random forest'])
n_components = 5



plot_data = pd.DataFrame(X_svd[:1000, :n_components], columns=['f{}'.format(i) for i in range(n_components)])

plot_data['target'] = target[:1000]

sns.pairplot(plot_data, hue='target')
# tokenize

import nltk



tok_corpus = []

for sent in tqdm(corpus):

#     tok_corpus.append(nltk.word_tokenize(sent))

    tok_corpus.append(sent.split())



tok_corpus[:5]
# count words

counter = {}

for sent in tqdm(tok_corpus):

    for word in sent:

        counter[word] = counter.get(word, 0) + 1


plt.plot(sorted([np.log10(v) for v in counter.values()], reverse=True))

plt.xlabel('Word id')

plt.ylabel('log(frequency)')
# Make vocab

vocab = sorted(list(counter), key=counter.get,reverse=True)

print('Length:', len(vocab))

# Take only frequent words

min_count = 10

vocab = [word for word in vocab if counter[word] >= min_count]

print('Length:', len(vocab))

# Add <UNK> token

vocab.append('<UNK>')

print(vocab[:5])



# Make word index

word2idx = {word: idx for (idx, word) in enumerate(vocab)}

print(word2idx['the'], word2idx['to'], word2idx['hello'])
encoded_corpus = [[word2idx[word] for word in sent if word in word2idx] for sent in tqdm(tok_corpus)]

encoded_corpus[:5]
import gensim



w2v_google = gensim.models.KeyedVectors.load_word2vec_format("../input/nlpword2vecembeddingspretrained/GoogleNews-vectors-negative300.bin", binary=True)
vec = w2v_google['hello']

print(type(vec))

print(vec.shape)
w2v_google.most_similar([vec])
w2v_google.most_similar(['hello'])
w2v_google.most_similar(positive=['woman', 'king'], negative=['man'])
w2v_google.most_similar(positive=['father', 'woman'], negative=['man'])
# YOUR CODE HERE

# w2v_google.most_similar(positive=['...', '...'], negative=['...'])
w2v_google.most_similar(['fuck'])
w2v_google.most_similar(['ass'])
# Bag of words

X_w2v = [np.sum([np.zeros(vec.shape)] + [w2v_google[w] for w in sent if w in w2v_google], axis=0) for sent in tqdm(tok_corpus)]

normalize = lambda x: x / np.sqrt(np.sum(x**2) + 1e-8)

X_w2v = [normalize(x) for x in tqdm(X_w2v)]

X_w2v = np.array(X_w2v)

X_w2v.shape
%time eval_on_trainset(X_w2v[:1000], target[:1000])
%time eval_on_trainset(X_w2v[:10000], target[:10000], ['Log regression', 'random forest'])
n_components = 5



plot_data = pd.DataFrame(X_w2v[:1000, :n_components], columns=['f{}'.format(i) for i in range(n_components)])

plot_data['target'] = target[:1000]

sns.pairplot(plot_data, hue='target')
n_components = 5



plot_data = pd.DataFrame(TruncatedSVD(n_components=n_components).fit_transform(X_w2v)[:1000], columns=['f{}'.format(i) for i in range(n_components)])

plot_data['target'] = target[:1000]

sns.pairplot(plot_data, hue='target')
from sklearn.cluster import MiniBatchKMeans, MeanShift, AgglomerativeClustering, DBSCAN

from sklearn.mixture import GaussianMixture



from sklearn.metrics import silhouette_score
def distance(point1, point2):

    return np.sqrt(np.sum((point1 - point2)**2))



dists = []

scores = []



for k in tqdm(range(2, 21)):

    kmeans = MiniBatchKMeans(k)

    kmeans.fit(X_w2v)

    centers = kmeans.cluster_centers_

    labels = kmeans.predict(X_w2v)

    # Mean squared distance

    mean_dist = np.sum([distance(x, centers[label])**2 for x, label in zip(X_w2v, labels)])

    dists.append(mean_dist)

    # Silhouette

    score = silhouette_score(X_w2v[:2000], labels[:2000], metric='euclidean')

    scores.append(score)
plt.figure(figsize=(14, 8))

plt.plot(np.arange(2, 21), dists)

plt.ylabel('Sum of squared distance')

plt.xlabel('Number of clusters k')

plt.show()
plt.figure(figsize=(14, 8))

plt.plot(np.arange(2, 21), scores)

plt.ylabel('Silhouette score')

plt.xlabel('Number of clusters k')

plt.show()
k = 7

kmeans = MiniBatchKMeans(k, random_state=2)

kmeans.fit(X_w2v)

train['KMeans'] = kmeans.predict(X_w2v)
kmeans.predict(X_w2v)
plt.subplots(figsize=(10,6))

sns.barplot(x='KMeans' , y='toxic' , data=train)

plt.ylabel("Toxic")

plt.title("Toxic as function of KMeans")

plt.show()
# Find the most toxic sentence

am = np.argmin(np.sum((X_w2v - kmeans.cluster_centers_[0].reshape([1, 300]))**2, axis=1))

corpus[am]
# Find the least toxic sentence

am = np.argmin(np.sum((X_w2v[target == 1] - kmeans.cluster_centers_[1].reshape([1, 300]))**2, 1))

corpus[target == 1].iloc[am]
k = 7

kmeans = MiniBatchKMeans(k, random_state=0)

kmeans.fit(X_svd)

train['KMeans_SVD'] = kmeans.predict(X_svd)
plt.subplots(figsize=(10,6))

sns.barplot(x='KMeans_SVD' , y='toxic' , data=train)

plt.ylabel("Toxic")

plt.title("Toxic as function of KMeans")

plt.show()
# Find the most toxic sentence

am = np.argmin(np.sum((X_svd - kmeans.cluster_centers_[0].reshape([1, 100]))**2, 1))

corpus[am]
preds = (train['KMeans'] == 0).astype('int')

print('accuracy', accuracy_score(preds, target))

print('precision', precision_score(target, preds))

print('recall', recall_score(target, preds))

print('f1', f1_score(target, preds))
train.head(20)
%%time

meanshift = MeanShift(bandwidth=0.9)

meanshift.fit(X_w2v[:1000])
train['MeanShift'] = meanshift.predict(X_w2v)

print(len(train['MeanShift'].unique()))
plt.subplots(figsize=(10,6))

sns.barplot(x='MeanShift' , y='toxic' , data=train)

plt.ylabel("Toxic")

plt.title("Toxic as function of Mean Shift")

plt.show()
preds = (train['MeanShift'] == 10).astype('int')

print('accuracy', accuracy_score(preds, target))

print('precision', precision_score(target, preds))

print('recall', recall_score(target, preds))

print('f1', f1_score(target, preds))
k = 7

gmix = GaussianMixture(k, random_state=2)

%time gmix.fit(X_w2v[:10000])

train['GMixture'] = gmix.predict(X_w2v)
plt.subplots(figsize=(10,6))

sns.barplot(x='GMixture' , y='toxic' , data=train)

plt.ylabel("Toxic")

plt.title("Toxic as function of GMixture")

plt.show()
preds = (train['GMixture'] == 1).astype('int')

print('accuracy', accuracy_score(preds, target))

print('precision', precision_score(target, preds))

print('recall', recall_score(target, preds))

print('f1', f1_score(target, preds))
train.head()
n_cluterizers = 4



train_cl = train.iloc[:, -n_cluterizers:]

for cl in train_cl.columns:

    print(cl, train_cl[cl].unique())

    for i in sorted(train_cl[cl].unique()):

        train_cl[cl + '_%i' % i] = (train_cl[cl] == i).astype(int)

train_cl = train_cl.iloc[:, n_cluterizers:]

train_cl.head()
%time eval_on_trainset(train_cl.iloc[:10000, :], target[:10000])