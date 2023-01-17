# help()
# import platform  
# platform.architecture()  

# I met "Memory error so that I want to see the version and the bit imfo"
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

file = "../input/bag-of-words-meets-bags-of-popcorn-/labeledTrainData.tsv"
data_train = pd.read_csv(file, sep="\t")
file = "../input/bag-of-words-meets-bags-of-popcorn-/testData.tsv"
data_test = pd.read_csv(file, sep="\t")
file = "../input/bag-of-words-meets-bags-of-popcorn-/unlabeledTrainData.tsv"
data_unlabeled = pd.read_csv(file, sep="\t", error_bad_lines=False)
# the difference between .csv and .tsv is taht tsv file use "\t" as sep rather than ","

y_train = np.array(data_train.iloc[:,1])
data_train.drop(["sentiment"], axis = 1, inplace = True)
x_train = np.array(data_train)
x_test = np.array(data_test)
x_unlabeled = np.array(data_unlabeled)

n_samples_train = x_train.shape[0]
n_samples_test = x_test.shape[0]
n_samples_unlabeled = x_unlabeled.shape[0]

print(x_train.shape)
print(x_test.shape)
print(x_unlabeled.shape)
print(x_train[0])
# Any results you write to the current directory are saved as output.
from bs4 import BeautifulSoup
from nltk.stem.porter import PorterStemmer  
import re
def scanner_stemmer(review):
    text = BeautifulSoup(review, "html.parser").get_text()
    text = re.sub("[^a-zA-Z]"," ", text)
    text = text.split()
    porter_stemmer = PorterStemmer()
    for index,item in enumerate(text):
        text[index] = porter_stemmer.stem(item)
    text = " ".join(text)
    return text
import nltk.data
# nltk.download()

def review2sentences( review, tokenizer, remove_stopwords=False ):
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    raw_sentences = tokenizer.tokenize(review.strip())
    sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            sentences.append( review_to_wordlist( raw_sentence, remove_stopwords ))
    return sentences

def reviews2sentences(x_1, x_2):
    sentences = []  
    for review in x_1["review"]:
        sentences += review_to_sentences(review, tokenizer)
    for review in x_2["review"]:
        sentences += review_to_sentences(review, tokenizer)
    print(len(sentences))
from sklearn.feature_extraction.text import CountVectorizer as CV
from sklearn.feature_extraction.text import TfidfVectorizer as TFIDF
# TF-IDF will convert all characters to lowercase by default

def tfidf(x_1, x_2):
    n_samples_1 = len(x_1)
    n_samples_2 = len(x_2)
    
    # vectorizer = CV(analyzer='word', ngram_range=(1, 2))
    vectorizer = TFIDF(max_features=40000,sublinear_tf=1,analyzer='word', ngram_range=(1, 3)) #sublinear_tf=1

    x_all = x_1 + x_2
    x_all = vectorizer.fit_transform(x_all)
    x_1 = x_all[:n_samples_1]
    x_2 = x_all[n_samples_1:]
    print(x_1[0].shape, "\n")
    return x_1, x_2
# x_sentence_all = reviews2sentences(x_train, x_unlabeled)

# num_features = 300    # Word vector dimensionality                      
# min_word_count = 40   # Minimum word count                        
# num_workers = 4       # Number of threads to run in parallel
# context = 10          # Context window size                                                                                    
# downsampling = 1e-3   # Downsample setting for frequent words

# from gensim.models import word2vec
# print "Training model..."
# model = word2vec.Word2Vec(sentences, workers=num_workers, size=num_features, min_count = min_word_count, 
#                           window = context, sample = downsampling)


# model.init_sims(replace=True)
# model_name = "300features_40minwords_10context"
# model.save(model_name)
import gensim  
model = gensim.models.KeyedVectors.load_word2vec_format('../input/googlenewsvectorsnegative300/GoogleNews-vectors-negative300.bin',binary=True)  
len_vec_review = 300
def reviews2vec(reviews):
    n_reviews = reviews.shape[0]
    vec_review = np.zeros((n_reviews, len_vec_review))
    index2word_set = set(model.index2word)
    for i in range(n_reviews):
        cnt = 0
        text = BeautifulSoup(reviews[i][1], "html.parser").get_text()
        text = re.sub("[^a-zA-Z]"," ", text)
        text = text.split()
        for word in text:
            if word in index2word_set:
                cnt += 1
                vec_review[i] += model[word]
        vec_review[i] /= cnt
    return vec_review
# reviews to text after being lowered case and stemmed.
# x_text_train is a list, each obj is a string.
x_text_train = []
for i in range(n_samples_train):
    x_text_train.append(scanner_stemmer(x_train[i][1]))
print(x_text_train[0])

x_text_test = []
for i in range(n_samples_test):
    x_text_test.append(scanner_stemmer(x_test[i][1]))

## text to vector
x_vec_train_tfidf, x_vec_train_tfidf = tfidf(x_text_train, x_text_test)
    
# reviews to vector 
x_vec_train_w2v = reviews2vec(x_train)
x_vec_test_w2v = reviews2vec(x_test)
print(x_vec_train_w2v[0])
print(model["moment"])
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score

def general_function(mod_name, model_name, x_train, y_train, x_test):
    y_pred = model_train_predict(mod_name, model_name, x_train, y_train, x_test)
    output_prediction(y_pred, model_name)

def get_score(clf, x_train, y_train, use_acc=True):
    if use_acc:
        y_pred = clf.predict(x_train)
        right_num = (y_train == y_pred).sum()
        print("acc: ", right_num/n_samples_train)
        scores = cross_val_score(clf, x_train, y_train, cv=5)
        print("K-fold Accuracy: %0.3f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    else:
        y_pred = clf.predict_proba(x_train)[:,1]
        score_auc = roc_auc_score(y_train, y_pred)
        print("auc: ", score_auc)
        scores = cross_val_score(clf, x_train, y_train, cv=5, scoring='roc_auc')
        print("K-fold AUC: %0.3f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    

def model_train_predict(mod_name, model_name, x_train, y_train, x_test, use_acc=True):
    import_mod = __import__(mod_name, fromlist = str(True))
    if hasattr(import_mod, model_name):
         f = getattr(import_mod, model_name)
    else:
        print("404")
        return []
    clf = f()
    clf.fit(x_train, y_train)
    get_score(clf, x_train, y_train, use_acc = False)
    if use_acc:
        y_pred = clf.predict(x_test)
    else:
        y_pred = clf.predict_proba(x_test)[:,1]
    return y_pred

def output_prediction(y_pred, model_name):
    print(y_pred)
    data_predict = {"id":x_test[:,0], "sentiment":y_pred}
    data_predict = pd.DataFrame(data_predict)
    data_predict.to_csv("bwmbp output %s.csv" %model_name, index = False)
# from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
mod_name = "sklearn.naive_bayes"
# model_name = "GaussianNB"
model_name = "MultinomialNB"
# model_name = "BernoulliNB"
general_function(mod_name, model_name, x_vec_train, y_train, x_vec_test)
# from sklearn.svm import SVC
mod_name = "sklearn.svm"
model_name = "SVC"
# general_function(mod_name, model_name, x_vec_train, y_train, x_vec_test)
from sklearn.ensemble import RandomForestClassifier
mod_name = "sklearn.ensemble"
model_name = "RandomForestClassifier"
# general_function(mod_name, model_name, x_vec_train, y_train, x_vec_test)

clf = RandomForestClassifier(n_estimators=100, min_samples_split=50)
clf.fit(x_vec_train_w2v, y_train)
get_score(clf, x_vec_train_w2v, y_train, use_acc = False)
y_pred = clf.predict_proba(x_vec_test_w2v)[:,1]
output_prediction(y_pred, model_name)
from sklearn.linear_model import LogisticRegression
mod_name = "sklearn.linear_model"
model_name = "LogisticRegression"
general_function(mod_name, model_name, x_vec_train_w2v, y_train, x_vec_test_w2v)