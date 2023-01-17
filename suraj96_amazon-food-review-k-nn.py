import warnings
warnings.filterwarnings("ignore")

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import sqlite3
import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

from gensim.models import Word2Vec

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

from tqdm import tqdm

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

import os
print(os.listdir("../input"))


conn = sqlite3.connect('../input/database.sqlite')
filtered_data = pd.read_sql_query(""" SELECT * FROM Reviews WHERE Score != 3 """, conn)

def partition(x):
    if x < 3:
        return 0
    return 1

actualScore = filtered_data['Score']
positiveNegative = actualScore.map(partition) 
filtered_data['Score'] = positiveNegative
print("Number of data points in our data", filtered_data.shape)
filtered_data.head(5)
display= pd.read_sql_query("""
SELECT *
FROM Reviews
WHERE Score != 3 AND UserId="AR5J8UI46CURR"
ORDER BY ProductID
""", conn)
display.head()
sorted_data=filtered_data.sort_values('ProductId', axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last')

final=sorted_data.drop_duplicates(subset={"UserId","ProfileName","Time","Text"}, keep='first', inplace=False)

(final['Id'].size*1.0)/(filtered_data['Id'].size*1.0)*100
display= pd.read_sql_query("""
SELECT *
FROM Reviews
WHERE Score != 3 AND Id=44737 OR Id=64422
ORDER BY ProductID
""", conn)

display.head()
final=final[final.HelpfulnessNumerator<=final.HelpfulnessDenominator]

print(final.shape)
final['Score'].value_counts()
stop = set(stopwords.words('english'))
sno = nltk.stem.SnowballStemmer('english')

def cleanhtml(sentence):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ', sentence)
    return cleantext
def cleanpunc(sentence):
    cleaned = re.sub(r'[?|!|\'|"|#]',r'',sentence)
    cleaned = re.sub(r'[.|,|)|(|\|/]',r' ',cleaned)
    return cleaned
i=0
str1=' '
final_string=[]
all_positive_words=[] 
all_negative_words=[] 
s=''
for sent in tqdm(final['Text'].values):
    filtered_sentence=[]
    sent=cleanhtml(sent) 
    for w in sent.split():
        for cleaned_words in cleanpunc(w).split():
            if((cleaned_words.isalpha()) & (len(cleaned_words)>2)):    
                if(cleaned_words.lower() not in stop):
                    s=(sno.stem(cleaned_words.lower())).encode('utf8')
                    filtered_sentence.append(s)
                    if (final['Score'].values)[i] == 1: 
                        all_positive_words.append(s) 
                    if(final['Score'].values)[i] == 0:
                        all_negative_words.append(s) 
                else:
                    continue
            else:
                continue 
    str1 = b" ".join(filtered_sentence) #final string of cleaned words

    final_string.append(str1)
    i+=1

final['CleanedText']= final_string
final['CleanedText']= final['CleanedText'].str.decode("utf-8")
data_pos = final[final['Score'] == 1].sample(n = 1000)
print('Shape of positive reviews', data_pos.shape)
print()

data_neg = final[final['Score'] == 0].sample(n = 1000)
print('Shape of negative reviews', data_neg.shape)
print()

final_reviews = pd.concat([data_pos, data_neg])
print('Shape of final reviews', final_reviews.shape)
final_reviews['Time'] = pd.to_datetime(final_reviews['Time'])
final_reviews = final_reviews.sort_values(by='Time', ascending=True)
final_reviews[0:5]
X = final_reviews['CleanedText']
y = final_reviews['Score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=0)
print('X_train, y_train', X_train.shape, y_train.shape)
print('X_test, y_test', X_test.shape, y_test.shape)
def k_classifier_brute(X_train, y_train):
    myList = list(range(0,50))
    neighbors = list(filter(lambda x: x % 2 != 0, myList))

    cv_scores = []

    for k in neighbors:
        knn = KNeighborsClassifier(n_neighbors=k, algorithm = "brute")
        scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
        cv_scores.append(scores.mean())

    MSE = [1 - x for x in cv_scores]

    optimal_k = neighbors[MSE.index(min(MSE))]
    print('\nThe optimal number of neighbors is %d.' % optimal_k)

    return optimal_k
def k_classifier_kd(X_train, y_train):
    myList = list(range(0,50))
    neighbors = list(filter(lambda x: x % 2 != 0, myList))

    cv_scores = []

    for k in neighbors:
        knn = KNeighborsClassifier(n_neighbors=k, algorithm = "kd_tree")
        scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
        cv_scores.append(scores.mean())

    MSE = [1 - x for x in cv_scores]

    optimal_k = neighbors[MSE.index(min(MSE))]
    print('\nThe optimal number of neighbors is %d.' % optimal_k)

    return optimal_k
bow_xtrain = X_train
bow_xtest = X_test
bow_ytrain = y_train
bow_ytest = y_test

count_vect = CountVectorizer(ngram_range=(1,1))
bow_xtrain = count_vect.fit_transform(bow_xtrain)
bow_xtest = count_vect.transform(bow_xtest)

print(bow_xtrain.shape)
print(bow_xtest.shape)
k_brute = k_classifier_brute(bow_xtrain, bow_ytrain)
knn_bow_brute = KNeighborsClassifier(n_neighbors=k_brute)
knn_bow_brute.fit(bow_xtrain, bow_ytrain)
pred1 = knn_bow_brute.predict(bow_xtest)
print('Accuracy on test data using brute force', accuracy_score(bow_ytest, pred1))

print('*************************************')

k_kdtree = k_classifier_kd(bow_xtrain, bow_ytrain)
knn_bow_kdtree = KNeighborsClassifier(n_neighbors=k_kdtree)
knn_bow_kdtree.fit(bow_xtrain, bow_ytrain)
pred2 = knn_bow_kdtree.predict(bow_xtest)
print('Accuracy on test data using kdtree', accuracy_score(bow_ytest, pred2))
tfidf_xtrain = X_train
tfidf_xtest = X_test
tfidf_ytrain = y_train
tfidf_ytest = y_test

tf_idf_vect = TfidfVectorizer(ngram_range=(1,2))
tfidf_xtrain = tf_idf_vect.fit_transform(tfidf_xtrain)
tfidf_xtest = tf_idf_vect.transform(tfidf_xtest)
k_brute = k_classifier_brute(tfidf_xtrain, tfidf_ytrain)
knn_tfidf_brute = KNeighborsClassifier(n_neighbors=k_brute)
knn_tfidf_brute.fit(tfidf_xtrain, tfidf_ytrain)
pred1 = knn_tfidf_brute.predict(tfidf_xtest)
print('Accuracy on test data using brute force', accuracy_score(tfidf_ytest, pred1))

print('*************************************')

k_kdtree = k_classifier_kd(tfidf_xtrain, tfidf_ytrain)
knn_tfidf_kdtree = KNeighborsClassifier(n_neighbors=k_kdtree)
knn_tfidf_kdtree.fit(tfidf_xtrain, tfidf_ytrain)
pred2 = knn_tfidf_kdtree.predict(tfidf_xtest)
print('Accuracy on test data using kdtree', accuracy_score(tfidf_ytest, pred2))
# W2V using training data

list_of_sent_train=[]
for sent in X_train:
    list_of_sent_train.append(sent.split())
    
w2v_model_train = Word2Vec(list_of_sent_train, min_count=5, size=50, workers=4)
w2v_words_train = w2v_model_train[w2v_model_train.wv.vocab]
# W2V using test data
list_of_sent_test=[]
for sent in X_test:
    list_of_sent_test.append(sent.split())
    
w2v_model_test = Word2Vec(list_of_sent_test, min_count=5, size=50, workers=4)
w2v_words_test = w2v_model_test[w2v_model_test.wv.vocab]
# Avg W2V for training data
sent_vectors_train = []
for sent in list_of_sent_train:
    sent_vec = np.zeros(50) 
    cnt_words =0
    for word in sent:
        try:
            vec = w2v_model_train.wv[word]
            sent_vec += vec
            cnt_words += 1
        except:
            pass
    sent_vec /= cnt_words
    sent_vectors_train.append(sent_vec)
# Avg W2V for testing data
sent_vectors_test = []
for sent in list_of_sent_test:
    sent_vec = np.zeros(50) 
    cnt_words =0
    for word in sent:
        try:
            vec = w2v_model_test.wv[word]
            sent_vec += vec
            cnt_words += 1
        except:
            pass
    sent_vec /= cnt_words
    sent_vectors_test.append(sent_vec)
w2v_xtrain = sent_vectors_train
w2v_xtest = sent_vectors_test
w2v_ytrain = y_train
w2v_ytest = y_test

k_brute = k_classifier_brute(w2v_xtrain, w2v_ytrain)
knn_w2v_brute = KNeighborsClassifier(n_neighbors=k_brute)
knn_w2v_brute.fit(w2v_xtrain, w2v_ytrain)
pred1 = knn_w2v_brute.predict(w2v_xtest)
print('Accuracy on test data using brute force', accuracy_score(w2v_ytest, pred1))


print('*************************************')


k_kdtree = k_classifier_kd(w2v_xtrain, w2v_ytrain)
knn_w2v_kdtree = KNeighborsClassifier(n_neighbors=k_kdtree)
knn_w2v_kdtree.fit(w2v_xtrain, w2v_ytrain)
pred2 = knn_w2v_kdtree.predict(w2v_xtest)
print('Accuracy on test data using kdtree', accuracy_score(w2v_ytest, pred2))
# Using training data
model1 = TfidfVectorizer()
tf_idf_matrix = model1.fit_transform(X_train.values)
dictionary = dict(zip(model1.get_feature_names(), list(model1.idf_)))

tfidf_feat = model1.get_feature_names()

tfidf_sent_vectors_train = []
row=0;
for sent in tqdm(list_of_sent_train):
    sent_vec = np.zeros(50) 
    weight_sum =0; 
    for word in sent: 
        if word in w2v_words_train:
            vec = w2v_model_train.wv[word]
            tf_idf = dictionary[word]*(sent.count(word)/len(sent))
            sent_vec += (vec * tf_idf)
            weight_sum += tf_idf
    if weight_sum != 0:
        sent_vec /= weight_sum
    tfidf_sent_vectors_train.append(sent_vec)
    row += 1
model2 = TfidfVectorizer()
tf_idf_matrix = model2.fit_transform(X_test.values)
dictionary = dict(zip(model2.get_feature_names(), list(model2.idf_)))

tfidf_feat = model2.get_feature_names()

tfidf_sent_vectors_test = []
row=0;
for sent in tqdm(list_of_sent_test):
    sent_vec = np.zeros(50) 
    weight_sum =0; 
    for word in sent: 
        if word in w2v_words_test:
            vec = w2v_model_test.wv[word]
            tf_idf = dictionary[word]*(sent.count(word)/len(sent))
            sent_vec += (vec * tf_idf)
            weight_sum += tf_idf
    if weight_sum != 0:
        sent_vec /= weight_sum
    tfidf_sent_vectors_test.append(sent_vec)
    row += 1
tfw2v_xtrain = tfidf_sent_vectors_train
tfw2v_xtest = tfidf_sent_vectors_test
tfw2v_ytrain = y_train
tfw2v_ytest = y_test

k_brute = k_classifier_brute(tfw2v_xtrain, tfw2v_ytrain)
knn_tfw2v_brute = KNeighborsClassifier(n_neighbors=k_brute)
knn_tfw2v_brute.fit(tfw2v_xtrain, tfw2v_ytrain)
pred1 = knn_tfw2v_brute.predict(tfw2v_xtest)
print('Accuracy on test data using brute force', accuracy_score(tfw2v_ytest, pred1))

print('*************************************')

k_kdtree = k_classifier_kd(tfw2v_xtrain, tfw2v_ytrain)
knn_tfw2v_kdtree = KNeighborsClassifier(n_neighbors=k_kdtree)
knn_tfw2v_kdtree.fit(tfw2v_xtrain, tfw2v_ytrain)
pred2 = knn_tfw2v_kdtree.predict(tfw2v_xtest)
print('Accuracy on test data using kdtree', accuracy_score(tfw2v_ytest, pred2))
