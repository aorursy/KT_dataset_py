# load packages

import glob

from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer,HashingVectorizer

from sklearn import decomposition, ensemble

from sklearn.decomposition import PCA

import xgboost



import keras

from keras.layers import Input, Lambda, Dense

from keras.models import Model

import keras.backend as K



from sklearn.utils import shuffle

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

import string

import pandas as pd

from nltk.stem.porter import PorterStemmer

import re

import spacy

from nltk.corpus import stopwords

from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS

from spacy.lang.en import English

spacy.load('en')

parser = English()
#Load text files from train folder

pos_tran_files = glob.glob('/kaggle/input/aclimdb/aclImdb/train/pos' + "/*.txt")

neg_train_files = glob.glob('/kaggle/input/aclimdb/aclImdb/train/neg' + "/*.txt")
#read text files from train folder

pos_train_txt = []

pos_train_label = []

for txt in pos_tran_files:

    data = open(txt,encoding="utf-8").read()

    pos_train_txt.append(data)

    pos_train_label.append('pos')

    

neg_train_txt = []

neg_train_label = []

for txt in neg_train_files:

    data = open(txt,encoding="utf-8").read()

    neg_train_txt.append(data)

    neg_train_label.append('neg')
#Load text files from test folder

pos_test_files = glob.glob('/kaggle/input/aclimdb/aclImdb/test/pos' + "/*.txt")

neg_test_files = glob.glob('/kaggle/input/aclimdb/aclImdb/test/neg' + "/*.txt")
#read text files from test folder

pos_test_txt = []

pos_test_label = []

for txt in pos_test_files:

    data = open(txt,encoding="utf-8").read()

    pos_test_txt.append(data)

    pos_test_label.append('pos')

    

neg_test_txt = []

neg_test_label = []

for txt in neg_test_files:

    data = open(txt,encoding="utf-8").read()

    neg_test_txt.append(data)

    neg_test_label.append('neg')
# create train dataframe

train_pos_DF = pd.DataFrame()



train_pos_DF['text'] = pos_train_txt

train_pos_DF['label'] = pos_train_label



train_neg_DF = pd.DataFrame()



train_neg_DF['text'] = neg_train_txt

train_neg_DF['label'] = neg_train_label



trainDF = pd.concat([train_pos_DF,train_neg_DF])

trainDF = shuffle(trainDF)

trainDF = trainDF.reset_index(drop=True)
# create test dataframe

test_pos_DF = pd.DataFrame()



test_pos_DF['text'] = pos_test_txt

test_pos_DF['label'] = pos_test_label



test_neg_DF = pd.DataFrame()



test_neg_DF['text'] = neg_test_txt

test_neg_DF['label'] = neg_test_label



testDF = pd.concat([test_pos_DF,test_neg_DF])

testDF = shuffle(testDF)

testDF = testDF.reset_index(drop=True)
trainDF.head()
trainDF.shape
testDF.head()
testDF.shape
trainDF['label'].unique()
trainDF['label'].value_counts()
sns.set(rc={'figure.figsize':(10,10)})

sns.countplot(trainDF['label'])
# stop words and spcecial characters 

STOPLIST = set(stopwords.words('english') + list(ENGLISH_STOP_WORDS)) 

SYMBOLS = " ".join(string.punctuation).split(" ") + ["-", "...", "”", "”","''"]



# Data Cleaner and tokenizer

def tokenizeText(text):

    

    text = text.strip().replace("\n", " ").replace("\r", " ")

    text = text.lower()

    

    tokens = parser(text)

    

    lemmas = []

    for tok in tokens:

        lemmas.append(tok.lemma_.lower().strip() if tok.lemma_ != "-PRON-" else tok.lower_)

    tokens = lemmas

    

    # reomve stop words and special charaters

    tokens = [tok for tok in tokens if tok.lower() not in STOPLIST]

    tokens = [tok for tok in tokens if tok not in SYMBOLS]

    

    tokens = [tok for tok in tokens if len(tok) >= 3]

    

    # remove remaining tokens that are not alphabetic

    tokens = [tok for tok in tokens if tok.isalpha()]

    

    # stemming of words

    #porter = PorterStemmer()

    #tokens = [porter.stem(word) for word in tokens]

    

    tokens = list(set(tokens))

    #return tokens

    return ' '.join(tokens[:])
# Data cleaning

trainDF['text'] = trainDF['text'].apply(lambda x:tokenizeText(x))

testDF['text'] = testDF['text'].apply(lambda x:tokenizeText(x))
# Data preparation

y_train = trainDF['label'].tolist()

x_train = trainDF['text'].tolist()



y_test = trainDF['label'].tolist()

x_test = trainDF['text'].tolist()
# Count Vectors as features

# create a count vectorizer object 

count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')

count_vect.fit(x_train+x_test)



# transform the training and test data using count vectorizer object

xtrain_count =  count_vect.transform(x_train)

xtest_count =  count_vect.transform(x_test)
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)

tfidf_vect.fit(x_train+x_test)

xtrain_tfidf =  tfidf_vect.transform(x_train)

xtest_tfidf =  tfidf_vect.transform(x_test)
tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)

tfidf_vect_ngram.fit(x_train+x_test)

xtrain_tfidf_ngram =  tfidf_vect_ngram.transform(x_train)

xtest_tfidf_ngram =  tfidf_vect_ngram.transform(x_test)
tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)

tfidf_vect_ngram_chars.fit(x_train+x_test)

xtrain_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(x_train) 

xtest_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(x_test) 
# getting train features

hash_vectorizer = HashingVectorizer(n_features=5000)

hash_vectorizer.fit(x_train+x_test)

xtrain_hash_vectorizer =  hash_vectorizer.transform(x_train) 

xtest_hash_vectorizer =  hash_vectorizer.transform(x_test)
def train_model(classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):

    # fit the training dataset on the classifier

    classifier.fit(feature_vector_train, label)

    

    # predict the labels on validation dataset

    predictions = classifier.predict(feature_vector_valid)

    

    return metrics.accuracy_score(predictions, y_test)
# Naive Bayes on Count Vectors

accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_count, y_train, xtest_count)

print("NB, Count Vectors: ", accuracy)



# Naive Bayes on Word Level TF IDF Vectors

accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf, y_train, xtest_tfidf)

print("NB, WordLevel TF-IDF: ", accuracy)



# Naive Bayes on Ngram Level TF IDF Vectors

accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram, y_train, xtest_tfidf_ngram)

print("NB, N-Gram Vectors: ", accuracy)



# Naive Bayes on Character Level TF IDF Vectors

accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram_chars, y_train, xtest_tfidf_ngram_chars)

print("NB, CharLevel Vectors: ", accuracy)
# Linear Classifier on Count Vectors

accuracy = train_model(linear_model.LogisticRegression(solver="lbfgs",multi_class="auto",max_iter=4000), xtrain_count, y_train, xtest_count)

print("LR, Count Vectors: ", accuracy)



# Linear Classifier on Word Level TF IDF Vectors

accuracy = train_model(linear_model.LogisticRegression(solver="lbfgs",multi_class="auto",max_iter=4000), xtrain_tfidf, y_train, xtest_tfidf)

print("LR, WordLevel TF-IDF: ", accuracy)



# Linear Classifier on Ngram Level TF IDF Vectors

accuracy = train_model(linear_model.LogisticRegression(solver="lbfgs",multi_class="auto",max_iter=4000), xtrain_tfidf_ngram, y_train, xtest_tfidf_ngram)

print("LR, N-Gram Vectors: ", accuracy)



# Linear Classifier on Character Level TF IDF Vectors

accuracy = train_model(linear_model.LogisticRegression(solver="lbfgs",multi_class="auto",max_iter=4000), xtrain_tfidf_ngram_chars, y_train, xtest_tfidf_ngram_chars)

print("LR, CharLevel Vectors: ", accuracy)



# Linear Classifier on Hash Vectors

accuracy = train_model(linear_model.LogisticRegression(solver="lbfgs",multi_class="auto",max_iter=4000), xtrain_hash_vectorizer, y_train, xtest_hash_vectorizer)

print("LR, Hash Vectors: ", accuracy)
# RF on Count Vectors

accuracy = train_model(ensemble.RandomForestClassifier(n_estimators=10), xtrain_count, y_train, xtest_count)

print("RF, Count Vectors: ", accuracy)



# RF on Word Level TF IDF Vectors

accuracy = train_model(ensemble.RandomForestClassifier(n_estimators=10), xtrain_tfidf, y_train, xtest_tfidf)

print("RF, WordLevel TF-IDF: ", accuracy)



# RF on Ngram Level TF IDF Vectors

accuracy = train_model(ensemble.RandomForestClassifier(n_estimators=10), xtrain_tfidf_ngram, y_train, xtest_tfidf_ngram)

print("RF, N-Gram Vectors: ", accuracy)



# RF on Character Level TF IDF Vectors

accuracy = train_model(ensemble.RandomForestClassifier(n_estimators=10), xtrain_tfidf_ngram_chars, y_train, xtest_tfidf_ngram_chars)

print("RF, CharLevel Vectors: ", accuracy)



# RF on Hash Vectors

accuracy = train_model(ensemble.RandomForestClassifier(n_estimators=10), xtrain_hash_vectorizer, y_train, xtest_hash_vectorizer)

print("RF, Hash Vectors: ", accuracy)
# Extereme Gradient Boosting on Count Vectorsy_train

accuracy = train_model(xgboost.XGBClassifier(), xtrain_count.tocsc(), y_train, xtest_count.tocsc())

print("Xgb, Count Vectors: ", accuracy)



# Extereme Gradient Boosting on Word Level TF IDF Vectors

accuracy = train_model(xgboost.XGBClassifier(), xtrain_tfidf.tocsc(), y_train, xtest_tfidf.tocsc())

print("Xgb, WordLevel TF-IDF: ", accuracy)



# Extereme Gradient Boosting on Ngram Level TF IDF Vectors

accuracy = train_model(xgboost.XGBClassifier(), xtrain_tfidf_ngram, y_train, xtest_tfidf_ngram)

print("Xgb, N-Gram Vectors: ", accuracy)



# Extereme Gradient Boosting on Character Level TF IDF Vectors

accuracy = train_model(xgboost.XGBClassifier(), xtrain_tfidf_ngram_chars.tocsc(), y_train, xtest_tfidf_ngram_chars.tocsc())

print("Xgb, CharLevel Vectors: ", accuracy)



# Extereme Gradient Boosting on Hash Vectors

accuracy = train_model(xgboost.XGBClassifier(), xtrain_hash_vectorizer, y_train, xtest_hash_vectorizer)

print("Xgb, Hash Vectors: ", accuracy)
# get elmo from tensorflow hub

import tensorflow_hub as hub

import tensorflow as tf



# ELMo Embedding

embed = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)



def ELMoEmbedding(x):

    return embed(tf.squeeze(tf.cast(x, tf.string)), signature="default", as_dict=True)["default"]
# Data preparation

le = preprocessing.LabelEncoder()

le.fit(y_train+y_test)



def encode(le, labels):

    enc = le.transform(labels)

    return keras.utils.to_categorical(enc)



def decode(le, one_hot):

    dec = np.argmax(one_hot, axis=1)

    return le.inverse_transform(dec)



y_train_enc = encode(le, y_train)

y_test_enc = encode(le, y_test)
# Build Model

input_text = Input(shape=(1,), dtype=tf.string)

embedding = Lambda(ELMoEmbedding, output_shape=(1024, ))(input_text)

dense = Dense(256, activation='relu')(embedding)

pred = Dense(2, activation='softmax')(dense)

model = Model(inputs=[input_text], outputs=pred)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



with tf.Session() as session:

    K.set_session(session)

    session.run(tf.global_variables_initializer())  

    session.run(tf.tables_initializer())

    history = model.fit(np.asarray(x_train), np.asarray(y_train_enc), epochs=10, batch_size=16)

    model.save_weights('./elmo-model.h5')

# Predict Test data

with tf.Session() as session:

    K.set_session(session)

    session.run(tf.global_variables_initializer())

    session.run(tf.tables_initializer())

    model.load_weights('./elmo-model.h5')  

    predicts = model.predict(np.asarray(x_test), batch_size=16)



y_test = decode(le, y_test_enc)

y_preds = decode(le, predicts)
from sklearn import metrics



print(metrics.confusion_matrix(y_test, y_preds))



print(metrics.classification_report(y_test, y_preds))



from sklearn.metrics import accuracy_score



print("Accuracy:",accuracy_score(y_test,y_preds))