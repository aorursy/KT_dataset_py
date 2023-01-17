import numpy as np

import pandas as pd

import random

import nltk

import re

import string

import matplotlib.pyplot as plt

from nltk.stem import PorterStemmer

from nltk.stem import WordNetLemmatizer

from nltk.tokenize import word_tokenize

from textblob import TextBlob

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn import decomposition

from sklearn import linear_model, naive_bayes, metrics, svm

import xgboost



def load_data():

    train = pd.read_csv('/kaggle/input/sentiment-analysis-on-movie-reviews/train.tsv', sep='\t', header = 0)

    train = np.array(train.values)

    train_data = train[:,0:3]

    train_label = train[:,-1]

    test = pd.read_csv('/kaggle/input/sentiment-analysis-on-movie-reviews/test.tsv', sep='\t', header = 0)

    test_data = np.array(test.values)

    return (train_data, train_label, test_data)

def get_count_features(data):

    pos_family = {

    'noun' : ['NN','NNS','NNP','NNPS'],

    'pron' : ['PRP','PRP$','WP','WP$'],

    'verb' : ['VB','VBD','VBG','VBN','VBP','VBZ'],

    'adj' :  ['JJ','JJR','JJS'],

    'adv' : ['RB','RBR','RBS','WRB']

    }

    line_tags=[]

    cnt_noun = cnt_pron = cnt_verb = cnt_adj = cnt_adv = 0

    cnt_features = []    

    for line in data:

        cnt_char = len(line)

        cnt_word = len(line.split())

        blob = TextBlob(line)

        tags = blob.tags

        for (word,tag) in tags:

            line_tags.append(tag)

            if tag in pos_family['noun']:

                cnt_noun += 1

            elif tag in pos_family['pron']:

                cnt_pron += 1

            elif tag in pos_family['verb']:

                cnt_verb += 1

            elif tag in pos_family['adj']:

                cnt_adj += 1

            elif tag in pos_family['adv']:

                cnt_adv += 1

        cnt_features.append([cnt_char, cnt_word, cnt_noun, cnt_pron, cnt_verb, cnt_adj, cnt_adv])

    return np.array(cnt_features)

def pre_process_data(reviews):

    reviews_clean = []

    line_clean=[]

    stemmer= PorterStemmer()

    lemmatizer=WordNetLemmatizer()



    for line in reviews:

        line = line.lower().strip()

        #print("lower:", line)

        line = re.sub(r'\d+', '', line)

        #print("no number:", line)

        line = line.translate(str.maketrans("","", string.punctuation))

        #print("no biaodian:", line)

        line = word_tokenize(line)

        for word in line:

            word = lemmatizer.lemmatize(word)

            word = stemmer.stem(word)

            line_clean.append(word)

        reviews_clean.append(line_clean) 

    return reviews_clean
def count_ngram_features(train_data, test_data):

    count_vec = CountVectorizer(analyzer='word', stop_words='english', ngram_range=(1,4),max_features=1000)

    train_count_ngram = count_vec.fit_transform(train_data)

    test_count_ngram = count_vec.transform(test_data)

    return (train_count_ngram, test_count_ngram)
def tfidf_ngram_features(train_data, test_data):

    tfidf_vec = TfidfVectorizer(analyzer='word', stop_words='english', ngram_range=(1,4),max_features=1000)

    train_tfidf_ngram = tfidf_vec.fit_transform(train_data)

    test_tfidf_ngram = tfidf_vec.fit_transform(test_data)

    return (train_tfidf_ngram, test_tfidf_ngram)
def get_PCA_features(features):

    pca = decomposition.IncrementalPCA(n_components = None, batch_size=10)

    pca.partial_fit(features.toarray())

    number = pca.n_components_

    result = pca.transform(features.toarray())

    return (number, result)
def train_model(classifier, train_feature_vector, train_label, test_feature_vector):

    classifier.fit(train_feature_vector, train_label)

    predictions = classifier.predict(test_feature_vector)

    return predictions
def save_result(file_name, test_data, predictions):

    result = np.hstack((test_data[:,0], predictions))

    np.savetxt(file_name, result, delimiter=None)
train_data, train_label, test_data = load_data()

train_reviews = train_data[:,2]

test_reviews = test_data[:,2]



#print (train_data.shape)

train_data[:,2] = pre_process_data(train_reviews)

train_cnt_features = get_count_features(train_reviews)

test_cnt_features = get_count_features(test_reviews)

print (train_cnt_features.shape)

print (test_cnt_features.shape)



train_count_ngram, test_count_ngram = count_ngram_features(train_reviews, test_reviews)

train_count_ngram = train_count_ngram.toarray()

test_count_ngram = test_count_ngram.toarray()

print (train_count_ngram.shape)

print (test_count_ngram.shape)



number, PCAfeatures = get_PCA_features(train_count_ngram)

print (number)



train_feature_vector = np.hstack((train_count_ngram,train_cnt_features))

print(train_feature_vector.shape)



test_feature_vector = np.hstack((test_count_ngram,test_cnt_features))

print(test_feature_vector.shape)



###朴素贝叶斯

predictions_NB = train_model(naive_bayes.MultinomialNB(), train_feature_vector, train_label.astype('int'), test_feature_vector)

save_result = ('NB_predictions_count.csv', test_data, predictions_NB)



###逻辑回归

predictions_LR = train_model(linear_model.LogisticRegression(), train_feature_vector, train_label.astype('int'), test_feature_vector)

save_result = ('LR_predictions_count.csv', test_data, predictions_LR)



###SVM

predictions_SVM = train_model(svm.SVC(), train_feature_vector, train_label.astype('int'), test_feature_vector)

save_result = ('SVM_predictions_count.csv', test_data, predictions_SVM)



###Xgboost

predictions_Xgb = train_model(xgboost.XGBClassifier(), train_feature_vector.tocsc(), train_label.astype('int'), test_feature_vector.tocsc())

save_result = ('Xgb_predictions_count.csv', test_data, predictions_Xgb)