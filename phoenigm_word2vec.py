!pip install pymorphy2

import nltk as nltk
import pandas as pd
import pymorphy2
import tensorflow as tf
analyzer = pymorphy2.MorphAnalyzer()

def get_normal_form_of_single_text(text):
    normalized_text = ""
    tokens = nltk.word_tokenize(text)
    
    words_array = []
    
    for token in tokens:            
        words_array.append(analyzer.parse(token)[0].normal_form)
    return words_array
# df = pd.read_csv("../input/allreviews/all-reviews.csv", names=["name", "review", "label"], skiprows=lambda i: i % 10 != 0 or i == 0)
df = pd.read_csv("../input/allreviews/all-reviews.csv", names=["name", "review", "label"], skiprows=1)
df['label'] = df['label'].map(lambda x: "1" if x not in ["-1", "0", "1"] else x)

df['review'] = df['review'].map(lambda x: get_normal_form_of_single_text(x))
print(df['review'])
my_films = ["Побег из Шоушенка", "Поймай меня, если сможешь", "Престиж"]
my_reviews = df[df.name.isin(my_films)]
train_reviews = df[~df.name.isin(my_films)]
train_reviews['review'].head()
from gensim.test.utils import datapath
from gensim import utils            
import gensim.models

model = gensim.models.Word2Vec(sentences=train_reviews['review'], min_count=3, iter=250)

print(model.wv.most_similar(positive=['шедевр'], topn=5))
print(model.wv.most_similar(positive=['юмор'], topn=5))
print(model.wv.most_similar(positive=['три'], topn=5))
print(model.wv.most_similar(positive=['проблема'], topn=5))
print(model.wv.most_similar(positive=['слово'], topn=5))
import numpy as np

def make_feature_vec(words, model, num_features):
    feature_vec = np.zeros((num_features,),dtype="float32")
    nwords = 0
    index2word_set = model.wv.index2word

    for word in words:
        if word in index2word_set: 
            nwords = nwords + 1
            feature_vec = np.add(feature_vec,model[word])
    
    feature_vec = np.divide(feature_vec, nwords)
    return feature_vec


def get_avg_feature_vecs(reviews, model, num_features):
    counter = 0
    review_feature_vecs = np.zeros((len(reviews),num_features), dtype='float32')
    
    for review in reviews:
        review_feature_vecs[counter] = make_feature_vec(review, model, num_features)
        counter = counter + 1
    return review_feature_vecs
    
X_train = get_avg_feature_vecs(train_reviews['review'], model, 100)
X_test = get_avg_feature_vecs(my_reviews['review'], model, 100)

print(X_train)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support

clf = RandomForestClassifier(n_estimators = 100, max_depth=20, random_state=0)
clf.fit(X_train, train_reviews.label)
accuracy = clf.score(X_test, my_reviews.label)

rf_predicted = clf.predict(X_test)
print(accuracy)
rf_precision_recall_fscore = precision_recall_fscore_support(my_reviews.label.values, rf_predicted)
print(f"Precision(-1, 0, 1) = {rf_precision_recall_fscore[0]}")
print(f"Recall(-1, 0, 1) = {rf_precision_recall_fscore[1]}")
print(f"Fscore(-1, 0, 1) = {rf_precision_recall_fscore[2]}")