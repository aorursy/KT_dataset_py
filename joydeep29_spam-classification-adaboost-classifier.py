# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import itertools
import nltk
import re
import numpy as np
from nltk.stem.snowball import SnowballStemmer,PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
import math
import re, string
from numpy.random import multivariate_normal
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import Perceptron, SGDClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.externals import joblib
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.corpus import stopwords

# Defining runtime parameters
stemmer = SnowballStemmer("english")
feature = 1000
estimator_number = 1000
component = 30
def resample(mean, cov, number, tag):
    array = multivariate_normal(mean=mean, cov=cov, size=number)
    response_new = np.repeat(tag, number)
    return array, response_new
def ReplaceNumbertoWords(text):
    p = inflect.engine()
    s = text
    num = re.findall('\d+', s)
    l = text.split(" ")
    if len(num) >= 1:
        # listreplace = [p.number_to_words(n) for n in num]
        for n in num:
            replacenumlist = np.where(np.array(l) == n)[0].tolist()
            for j in replacenumlist:
                l[j] = p.number_to_words(n)
        l1 = ' '.join(l)
    else:
        l1 = s
    return l1
def transform(sentiment):
    if sentiment == 'spam':
        return 1
    elif sentiment == 'ham':
        return 0

def toLower(texts):
    tokens = sent_tokenize(texts)
    words = [w.lower() for w in tokens]
    return " ".join(words)

def removeStopWords(text):
    stop_words = set(stopwords.words('english'))
    tokens = sent_tokenize(text)
    words = [w.lower() for w in tokens]
    words = [w for w in words if not w in stop_words]
    return " ".join(words)


def Fit(modelname, name, docs_train, docs_test, y_train, y_test):
    model = modelname
    model.fit(docs_train, y_train)
    y_pred = model.predict(docs_test)
    y_train_pred = model.predict(docs_train)
    score = accuracy_score(y_test, y_pred)
    trainScore = accuracy_score(y_train, y_train_pred)
    c = confusion_matrix(y_test, y_pred)
    c = np.round(c / c.sum(1).astype(float).reshape(-1, 1), 3)
    print ("ModelName:", name, "TestScore:", score, "TrainScore:",trainScore)
    return [model,name, feature, component, estimator_number, c[0, 0], c[0, 1], c[1, 0], c[1, 1], score,trainScore]
def BiasedSamplesBinary(dataset,Responsename,category):
    X_1 = dataset[dataset[Responsename] == category]     #dataset containing category with max count
    X_2 = dataset[dataset[Responsename] != category]     #dataset containing category with min count
    X_1.index = range(0,X_1.shape[0])
    X_2.index = range(0,X_2.shape[0])
    smpl = np.random.choice(range(0,X_1.shape[0]),2*X_2.shape[0],replace=False)
    X = pd.concat([X_1.loc[smpl],X_2],axis=0)
    X = X.sample(frac=1)
    return X

def Bias_Sampling_Check(dataset,Responsename):
    # global biased_dataset
    if len(dataset[Responsename].unique()) == 2:
        max_category = np.argmax(dataset[Responsename].value_counts())
        ratio = dataset[Responsename].value_counts()[np.argmax(dataset[Responsename].value_counts())]/float(dataset[Responsename].value_counts()[np.argmin(dataset[Responsename].value_counts())])
        if ratio > 2.0:
            biased_dataset = BiasedSamplesBinary(dataset,Responsename,max_category)
            biased_dataset.reset_index(drop = True, inplace = True)
    return biased_dataset[[Responsename]].values,biased_dataset.drop(Responsename,1).values


tweet = pd.read_csv('../input/spam.csv', encoding='latin-1')
tweet = tweet[['v1','v2']]
tweet.columns = ['sentiment','text']
tweet['text']=[t.replace('@', 'at') for t in tweet['text'].values]
tweet['text']=[t.replace('$', '') for t in tweet['text'].values]
tweet['text']=[t.replace('%', ' percent') for t in tweet['text'].values]
tweet['text']=[t.replace('.', '') for t in tweet['text'].values]
tweet['text']=[t.replace(',', '') for t in tweet['text'].values]
tweet['text']=[t.replace('!', ' surprisesituationtag ') for t in tweet['text'].values]
tweet['text']=[t.replace('#', '') for t in tweet['text'].values]
tweet['text']=[re.sub(r'https?:\/\/.*\/\w*','',t) for t in tweet['text'].values]
tweet['text']=[re.sub(r'['+string.punctuation+']+', ' ',t) for t in tweet['text'].values]
tweet['text']=[re.sub(r'\$\w*','',t) for t in tweet['text'].values]
tweet['text']=[t.replace('/', '') for t in tweet['text'].values]
tweet['text']=[t.replace('$', ' dollar ') for t in tweet['text'].values]
tweet['text']=[t.replace("?", '') for t in tweet['text'].values]
tweet['text']=[t.replace("&", '') for t in tweet['text'].values]
tweet['text']=[t.replace('~', 'nearly ') for t in tweet['text'].values]
tweet['text']=[t.replace('+', ' grow up higher high ') for t in tweet['text'].values]
tweet['text']=[t.replace('-', ' decline down lower less low') for t in tweet['text'].values]
tweet['text']=[stemmer.stem(t) for t in tweet['text'].values]

Response_array=np.array([transform(t) for  t in tweet['sentiment'].values])

vectorizer = TfidfVectorizer(stop_words='english',smooth_idf=True, max_features = feature)
vectorizer.fit(tweet['text'].values)
corpus_vec = vectorizer.transform(tweet['text'].values).toarray()
docs_train, docs_test, y_train, y_test = train_test_split(corpus_vec, Response_array, test_size = 0.20)
model, name, feature, component, estimator_number, c00, c01, c10, c11, score,trainScore = Fit(AdaBoostClassifier(n_estimators=200,learning_rate=0.9), 'Adaboost Classification', docs_train, docs_test, y_train, y_test)
print( " Actual Ham - Predicted Ham",c00)
print( " Actual Ham - Predicted Spam",c01)
print( " Actual Spam - Predicted Ham",c10)
print( " Actual Spam - Predicted Spam",c11)
