import numpy as np

import pandas as pd

import re

import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB

from sklearn.model_selection import cross_val_score

from bs4 import BeautifulSoup 

import nltk

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer, SnowballStemmer
# load data

df = pd.read_csv('../input/labeledTrainData.tsv', sep='\t')

df.head()
xtrain = df.review

ytrain = df.sentiment
stop_words = set(stopwords.words('english'))
# clean documents

def clean_document(document, stop_words=False, stem=False, lemm=False):

    clean_doc = document.lower()

    # remove HTML tags

    clean_doc = BeautifulSoup(clean_doc).get_text()

    # remove non-letters except ! and '

    clean_doc = re.sub('[^a-zA-Z!\']', ' ', clean_doc)

    # map ! to exclamation_point

    clean_doc = re.sub(r'!', ' exclamation_point ', clean_doc)

    clean_doc = re.sub("'", ' ', clean_doc)

    # remove stopwords 

    if stop_words:

        clean_doc = ' '.join([w for w in clean_doc.split() if w not in stop_words])

    # word lemmetization

    if lemm:

        lemmatizer = WordNetLemmatizer()

        clean_doc = ' '.join([lemmatizer.lemmatize(w) for w in clean_doc.split()])

    if stem:

        stemmer = SnowballStemmer('english')

        clean_doc = ' '.join([stemmer.stem(w) for w in clean_doc.split()])

    

    return clean_doc.lower()

   

# clean train data

xtrain_after_clean = xtrain.apply(clean_document, stop_words=stop_words, stem=False, lemm=True)
countvectorizer = CountVectorizer()

xtrain_count = countvectorizer.fit_transform(xtrain.values)



# train with naive bayes

model_1 = MultinomialNB()

model_1.fit(xtrain_count, ytrain)

cv = cross_val_score(model_1, xtrain_count, ytrain, cv=5, scoring='f1')

print('cross validation f1-score: ',cv.mean())
# check word occurence in pos and neg reivews

xtrain_pos = xtrain_after_clean.values[np.where(ytrain == 1)]

xtrain_neg = xtrain_after_clean.values[np.where(ytrain == 0)]

pos_countvectorizer = CountVectorizer()

neg_countvectorizer = CountVectorizer()

xtrain_pos_count = pos_countvectorizer.fit_transform(xtrain_pos)

xtrain_neg_count = neg_countvectorizer.fit_transform(xtrain_neg)



pos_words = pos_countvectorizer.get_feature_names()

pos_words_occurence = np.sum(xtrain_pos_count.toarray(), axis=0)

df_pos = pd.DataFrame({'word':pos_words, 'pos_count':pos_words_occurence})



neg_words = neg_countvectorizer.get_feature_names()

neg_words_occurence = np.sum(xtrain_neg_count.toarray(), axis=0)

df_neg = pd.DataFrame({'word':neg_words, 'neg_count':neg_words_occurence})



df_word_compare = df_pos.merge(df_neg, how='outer', on='word').sort_values(by='pos_count', ascending=False)

df_word_compare.head()
# add ratio between word in pos review and neg review

df_word_compare['ratio'] = df_word_compare['pos_count'] / df_word_compare['neg_count']

# check important word in positive reviews

df_word_compare[(df_word_compare.pos_count > 1000) & (df_word_compare.ratio > 3)].sort_values(by='ratio', ascending=False)
# check important word in negative reviews

df_word_compare[(df_word_compare.neg_count > 1000) & (df_word_compare.ratio < 1/3)].sort_values(by='ratio', ascending=False)
tfidfvectorizer = TfidfVectorizer(

    ngram_range=(1,3), 

    min_df=2,

    max_df=0.1,

    max_features=300000,

    use_idf=True

)

xtrain_tfidf = tfidfvectorizer.fit_transform(xtrain_after_clean.values)
# tweak weights of important words

xtrain_tfidf[:, tfidfvectorizer.vocabulary_['perfect']] *= 4

xtrain_tfidf[:, tfidfvectorizer.vocabulary_['amazing']] *=4

xtrain_tfidf[:, tfidfvectorizer.vocabulary_['favorite']] *= 4

xtrain_tfidf[:, tfidfvectorizer.vocabulary_['loved']] *= 2

xtrain_tfidf[:, tfidfvectorizer.vocabulary_['poor']] *= 5

xtrain_tfidf[:, tfidfvectorizer.vocabulary_['supposed']] *= 3

xtrain_tfidf[:, tfidfvectorizer.vocabulary_['boring']] *= 3

xtrain_tfidf[:, tfidfvectorizer.vocabulary_['terrible']] *= 3

xtrain_tfidf[:, tfidfvectorizer.vocabulary_['horrible']] *= 2

xtrain_tfidf[:, tfidfvectorizer.vocabulary_['awful']] *= 3

xtrain_tfidf[:, tfidfvectorizer.vocabulary_['worst']] *= 3

xtrain_tfidf[:, tfidfvectorizer.vocabulary_['waste']] *= 1.8




model = MultinomialNB()

model.fit(xtrain_tfidf, ytrain)

cv = cross_val_score(model, xtrain_tfidf, ytrain, cv=5, scoring='f1')

print('cross validation f1-score: ',cv.mean())
# load test set

test = pd.read_csv('../input/testData.tsv', sep='\t')



test_data = test.review

clean_test = test_data.apply(clean_document, stop_words=stop_words, stem=False, lemm=True)



# transform

test_tfidf = tfidfvectorizer.transform(clean_test.values)

preds = model.predict(test_tfidf)
# save results

pd.DataFrame({'id':test.id, 'sentiment':preds}).to_csv('submission.csv', index=False, header=True)