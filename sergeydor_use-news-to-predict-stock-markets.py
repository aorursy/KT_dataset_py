import re
import nltk
import pandas as pd
import numpy as np

from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
english_stemmer=nltk.stem.SnowballStemmer('english')

from sklearn.feature_selection.univariate_selection import SelectKBest, chi2, f_classif
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.linear_model import SGDClassifier, SGDRegressor,LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import random
import itertools

import sys
import os
import argparse
from sklearn.pipeline import Pipeline
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
import six
from abc import ABCMeta
from scipy import sparse
from scipy.sparse import issparse
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_X_y, check_array
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.preprocessing import normalize, binarize, LabelBinarizer
from sklearn.svm import LinearSVC

from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Lambda
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, SimpleRNN, GRU
from keras.preprocessing.text import Tokenizer
from collections import defaultdict
from keras.layers.convolutional import Convolution1D
from keras import backend as K
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
%matplotlib inline
plt.style.use('ggplot')
import collections as col
data = pd.read_csv('../input/Combined_News_DJIA.csv')
train = data[data['Date'] < '2015-01-01']
test = data[data['Date'] > '2014-12-31']
djia = pd.read_csv('../input/upload_DJIA_table.csv')
data.head()
data_merged = djia.merge(data, on='Date')

data_merged['Open_shift'] = data_merged['Open'].shift(1)
data_merged = data_merged.dropna()

data_merged['Label_adj'] = (data_merged['Open_shift'] >= data_merged['Open']).astype('int')

data_merged = data_merged.drop(['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close', 'Label', 'Open_shift'], axis=1)

train = data_merged[data_merged['Date'] < '2015-01-01']
test = data_merged[data_merged['Date'] > '2014-12-31']
train
test
train.iloc[0]['Top1']
mean_len = []
for i in range(len(train)):
    mean_len.append(np.mean([len(i) for i in train.loc[:, 'Top1':'Top25'].iloc[i]]))
    
sns.distplot(mean_len, kde=False)
sns.distplot(train['Label_adj'])
sns.distplot(test['Label_adj'])
plt.show()
plt.plot(djia['Date'], djia['Adj Close'])
plt.show()
contractions = { 
"ain't": "am not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he would",
"he'd've": "he would have",
"he'll": "he will",
"he's": "he is",
"how'd": "how did",
"how'll": "how will",
"how's": "how is",
"i'd": "i would",
"i'll": "i will",
"i'm": "i am",
"i've": "i have",
"isn't": "is not",
"it'd": "it would",
"it'll": "it will",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"must've": "must have",
"mustn't": "must not",
"needn't": "need not",
"oughtn't": "ought not",
"shan't": "shall not",
"sha'n't": "shall not",
"she'd": "she would",
"she'll": "she will",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"that'd": "that would",
"that's": "that is",
"there'd": "there had",
"there's": "there is",
"they'd": "they would",
"they'll": "they will",
"they're": "they are",
"they've": "they have",
"wasn't": "was not",
"we'd": "we would",
"we'll": "we will",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what will",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"where'd": "where did",
"where's": "where is",
"who'll": "who will",
"who's": "who is",
"won't": "will not",
"wouldn't": "would not",
"you'd": "you would",
"you'll": "you will",
"you're": "you are"
}
def clean_text(text, remove_stopwords = True):
    
    text = text.lower()
    
    # Replace contractions with their longer forms 
    if True:
        text = text.split()
        new_text = []
        for word in text:
            if word in contractions:
                new_text.append(contractions[word])
            else:
                new_text.append(word)
        text = " ".join(new_text)
    
    # Format words and remove unwanted characters
    text = re.sub(r'&amp;', '', text) 
    text = re.sub(r'0,0', '00', text) 
    text = re.sub(r'[_"\-;%()|.,+&=*%.,!?:#@\[\]]', ' ', text)
    text = re.sub(r'\'', ' ', text)
    text = re.sub(r'\$', ' $ ', text)
    text = re.sub(r'u s ', ' united states ', text)
    text = re.sub(r'u n ', ' united nations ', text)
    text = re.sub(r'u k ', ' united kingdom ', text)
    text = re.sub(r'j k ', ' jk ', text)
    text = re.sub(r' s ', ' ', text)
    text = re.sub(r' yr ', ' year ', text)
    text = re.sub(r' l g b t ', ' lgbt ', text)
    text = re.sub(r'0km ', '0 km ', text)
    
    # Optionally, remove stop words
    if remove_stopwords:
        text = text.split()
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]
        text = " ".join(text)
    return text
trainheadlines = []
for row in range(0,len(train.index)):
    trainheadlines.append(clean_text(' '.join(str(x) for x in train.iloc[row,1:26])))
    
testheadlines = []
for row in range(0,len(test.index)):
    testheadlines.append(clean_text(' '.join(str(x) for x in test.iloc[row,2:27])))
tokens_all = []
for headlines in trainheadlines:
    tokens = nltk.word_tokenize(headlines)
    tokens_all += tokens
    
counter_train = col.Counter(tokens_all)

tokens_all = []
for headlines in testheadlines:
    tokens = nltk.word_tokenize(headlines)
    tokens_all += tokens
    
counter_test = col.Counter(tokens_all)
c_train = np.array(counter_train.most_common(20))
c_test = np.array(counter_test.most_common(20))

plt.barh(c_train[:, 0], c_train[:, 1].astype('int'))
plt.title("train most freq")
plt.show()
plt.barh(c_test[:, 0], c_test[:, 1].astype('int'))
plt.title("test most freq")
plt.show()
basicvectorizer = CountVectorizer()
basictrain = basicvectorizer.fit_transform(trainheadlines)
print(basictrain.shape)
train_label = train['Label_adj']
test_label = test['Label_adj']
basicmodel = LogisticRegression()
basicmodel = basicmodel.fit(basictrain, train_label)
basicmodel = LogisticRegression(C=1.0)
basicmodel = basicmodel.fit(basictrain, train_label)

basictest = basicvectorizer.transform(testheadlines)
preds1 = basicmodel.predict(basictest)
acc_test=accuracy_score(test_label, preds1)
acc_train=accuracy_score(train_label, basicmodel.predict(basictrain))

print('Logic Regression 1 accuracy test: ',acc_test )
print('Logic Regression 1 accuracy train: ',acc_train )
print('Logic Regression 1 accuracy test: ',acc_test )
print('Logic Regression 1 accuracy train: ',acc_train )
basicwords = basicvectorizer.get_feature_names()
basiccoeffs = basicmodel.coef_.tolist()[0]
coeffdf = pd.DataFrame({'Word' : basicwords, 
                        'Coefficient' : basiccoeffs})
coeffdf = coeffdf.sort_values(['Coefficient', 'Word'], ascending=[0, 1])
coeffdf.head(5)
coeffdf.tail(5)
advancedvectorizer = TfidfVectorizer( min_df=0.03, max_df=0.97, max_features = 200000, ngram_range = (2, 2))
advancedtrain = advancedvectorizer.fit_transform(trainheadlines)
print(advancedtrain.shape)
advancedmodel = LogisticRegression(C=0.25)
advancedmodel = advancedmodel.fit(advancedtrain, train_label)

advancedtest = advancedvectorizer.transform(testheadlines)
preds2 = advancedmodel.predict(advancedtest)
acc_test=accuracy_score(test_label, preds2)
acc_train=accuracy_score(train_label, advancedmodel.predict(advancedtrain))

print('Logic Regression 2 accuracy test: ',acc_test )
print('Logic Regression 2 accuracy train: ',acc_train )
advwords = advancedvectorizer.get_feature_names()
advcoeffs = advancedmodel.coef_.tolist()[0]
advcoeffdf = pd.DataFrame({'Words' : advwords, 
                        'Coefficient' : advcoeffs})
advcoeffdf = advcoeffdf.sort_values(['Coefficient', 'Words'], ascending=[0, 1])
advcoeffdf.head(5)
advcoeffdf.tail(5)
advancedvectorizer = TfidfVectorizer( min_df=0.0039, max_df=0.1, max_features = 200000, ngram_range = (3, 3))
advancedtrain = advancedvectorizer.fit_transform(trainheadlines)
print(advancedtrain.shape)
advancedmodel = LogisticRegression()
advancedmodel = advancedmodel.fit(advancedtrain, train_label)
advancedmodel = LogisticRegression(C=0.15)
advancedmodel = advancedmodel.fit(advancedtrain, train_label)

advancedtest = advancedvectorizer.transform(testheadlines)
preds3 = advancedmodel.predict(advancedtest)
acc_test=accuracy_score(test_label, preds3)
acc_train=accuracy_score(train_label, advancedmodel.predict(advancedtrain))

print('Logic Regression 2 accuracy test: ',acc_test )
print('Logic Regression 2 accuracy train: ',acc_train )
def show_profit(model, t):
    preds_proba = model.predict_proba(advancedtest)

    preds = (preds_proba[:, 1] > t) * 1 + (preds_proba[:, 1] < (1 - t)) * 0 + (preds_proba[:, 1] >= (1-t)) * (preds_proba[:, 1] <= t) * -1

    test_merged = test.merge(djia, on='Date')
    test_merged['preds'] = preds

    test_merged['Open_shift'] = test_merged['Open'].shift(1)
    test_merged = test_merged.dropna()
    test_merged['profit'] = (np.abs((test_merged['Open_shift'] - test_merged['Open'])) * (test_merged['Label_adj'] == test_merged['preds']) - np.abs((test_merged['Open_shift'] - test_merged['Open'])) * (test_merged['Label_adj'] != test_merged['preds'])) * (test_merged['preds'] != -1)
    # test_merged = test_merged.sort_values(by='Date', ascending=True)
    test_merged['cum_profit'] = test_merged['profit'].cumsum()

    plt.plot(test_merged['cum_profit'])
    plt.show()
show_profit(advancedmodel, 0.5)
advwords = advancedvectorizer.get_feature_names()
advcoeffs = advancedmodel.coef_.tolist()[0]
advcoeffdf = pd.DataFrame({'Words' : advwords, 
                        'Coefficient' : advcoeffs})
advcoeffdf = advcoeffdf.sort_values(['Coefficient', 'Words'], ascending=[0, 1])
advcoeffdf.head(5)
advcoeffdf.tail(5)
advancedvectorizer = TfidfVectorizer( min_df=0.1, max_df=0.7, max_features = 200000, ngram_range = (1, 2))
advancedtrain = advancedvectorizer.fit_transform(trainheadlines)
print(advancedtrain.shape)
advancedmodel = MultinomialNB(alpha=0.1)
advancedmodel = advancedmodel.fit(advancedtrain, train_label)

advancedtest = advancedvectorizer.transform(testheadlines)
preds4 = advancedmodel.predict(advancedtest)
acc_test=accuracy_score(test_label, preds4)
acc_train=accuracy_score(train_label, advancedmodel.predict(advancedtrain))

print('Logic Regression 2 accuracy test: ',acc_test )
print('Logic Regression 2 accuracy train: ',acc_train )
show_profit(advancedmodel, 0.5)
advwords = advancedvectorizer.get_feature_names()
advcoeffs = advancedmodel.coef_.tolist()[0]
advcoeffdf = pd.DataFrame({'Words' : advwords, 
                        'Coefficient' : advcoeffs})
advcoeffdf = advcoeffdf.sort_values(['Coefficient', 'Words'], ascending=[0, 1])
advcoeffdf.head(5)
advcoeffdf.tail(5)
advancedvectorizer = TfidfVectorizer( min_df=0.03, max_df=0.2, max_features = 200000, ngram_range = (2, 2))
advancedtrain = advancedvectorizer.fit_transform(trainheadlines)
print(advancedtrain.shape)
advancedmodel = MultinomialNB(alpha=0.0001)
advancedmodel = advancedmodel.fit(advancedtrain, train_label)

advancedtest = advancedvectorizer.transform(testheadlines)
preds5 = advancedmodel.predict(advancedtest)
acc_test=accuracy_score(test_label, preds5)
acc_train=accuracy_score(train_label, advancedmodel.predict(advancedtrain))

print('Logic Regression 2 accuracy test: ',acc_test )
print('Logic Regression 2 accuracy train: ',acc_train )
print('NBayes 2 accuracy: ', acc5)
advwords = advancedvectorizer.get_feature_names()
advcoeffs = advancedmodel.coef_.tolist()[0]
advcoeffdf = pd.DataFrame({'Words' : advwords, 
                        'Coefficient' : advcoeffs})
advcoeffdf = advcoeffdf.sort_values(['Coefficient', 'Words'], ascending=[0, 1])
advcoeffdf.head(5)
advcoeffdf.tail(5)
advancedvectorizer = TfidfVectorizer( min_df=0.01, max_df=0.99, max_features = 200000, ngram_range = (1, 1))
advancedtrain = advancedvectorizer.fit_transform(trainheadlines)
print(advancedtrain.shape)
advancedmodel = RandomForestClassifier(max_depth=3, n_estimators=200)
advancedmodel = advancedmodel.fit(advancedtrain, train_label)

advancedtest = advancedvectorizer.transform(testheadlines)
preds6 = advancedmodel.predict(advancedtest)
acc_test=accuracy_score(test_label, preds6)
acc_train=accuracy_score(train_label, advancedmodel.predict(advancedtrain))

print('RF 1 accuracy test: ',acc_test )
print('RF 1 accuracy train: ',acc_train )
show_profit(advancedmodel, 0.545)
advancedvectorizer = TfidfVectorizer( min_df=0.03, max_df=0.2, max_features = 200000, ngram_range = (1, 2))
advancedtrain = advancedvectorizer.fit_transform(trainheadlines)
print(advancedtrain.shape)
advancedmodel = RandomForestClassifier(max_depth=4)
advancedmodel = advancedmodel.fit(advancedtrain, train_label)

advancedtest = advancedvectorizer.transform(testheadlines)
preds7 = advancedmodel.predict(advancedtest)
acc_test=accuracy_score(test_label, preds7)
acc_train=accuracy_score(train_label, advancedmodel.predict(advancedtrain))

print('Logic Regression 2 accuracy test: ',acc_test )
print('Logic Regression 2 accuracy train: ',acc_train )
show_profit(advancedmodel, 0.53)
advancedvectorizer = TfidfVectorizer( min_df=0.1, max_df=0.9, max_features = 200000, ngram_range = (1, 1))
advancedtrain = advancedvectorizer.fit_transform(trainheadlines)
print(advancedtrain.shape)
advancedmodel = GradientBoostingClassifier(n_estimators=100, max_depth=2)
advancedmodel = advancedmodel.fit(advancedtrain, train_label)

advancedtest = advancedvectorizer.transform(testheadlines)
preds8 = advancedmodel.predict(advancedtest.toarray())
acc_test=accuracy_score(test_label, preds8)
acc_train=accuracy_score(train_label, advancedmodel.predict(advancedtrain))

print('GB 1 accuracy test: ',acc_test )
print('GB 1 accuracy train: ',acc_train )
show_profit(advancedmodel, 0.5)
advancedvectorizer = TfidfVectorizer( min_df=0.02, max_df=0.175, max_features = 200000, ngram_range = (2, 2))
advancedtrain = advancedvectorizer.fit_transform(trainheadlines)
print(advancedtrain.shape)
advancedmodel = GradientBoostingClassifier()
advancedmodel = advancedmodel.fit(advancedtrain, train["Label"])
testheadlines = []
for row in range(0,len(test.index)):
    testheadlines.append(' '.join(str(x) for x in test.iloc[row,2:27]))
advancedtest = advancedvectorizer.transform(testheadlines)
preds9 = advancedmodel.predict(advancedtest.toarray())
acc9 = accuracy_score(test['Label'], preds9)
print('GBM 2 accuracy: ', acc9)
advancedvectorizer = TfidfVectorizer( min_df=0.2, max_df=0.8, max_features = 200000, ngram_range = (1, 1))
advancedtrain = advancedvectorizer.fit_transform(trainheadlines)
print(advancedtrain.shape)
advancedmodel = SGDClassifier(loss='modified_huber', random_state=0, shuffle=True)
advancedmodel = advancedmodel.fit(advancedtrain, train_label)

advancedtest = advancedvectorizer.transform(testheadlines)
preds10 = advancedmodel.predict(advancedtest.toarray())
acc_test=accuracy_score(test_label, preds10)
acc_train=accuracy_score(train_label, advancedmodel.predict(advancedtrain))

print('SGD 1 accuracy test: ',acc_test )
print('SGD 1 accuracy train: ',acc_train )
print('SGDClassifier 1: ', acc10)
advancedvectorizer = TfidfVectorizer( min_df=0.03, max_df=0.2, max_features = 200000, ngram_range = (2, 2))
advancedtrain = advancedvectorizer.fit_transform(trainheadlines)
print(advancedtrain.shape)
advancedmodel = SGDClassifier(loss='modified_huber', random_state=0, shuffle=True, alpha=100)
advancedmodel = advancedmodel.fit(advancedtrain, train_label)

advancedtest = advancedvectorizer.transform(testheadlines)
preds11 = advancedmodel.predict(advancedtest.toarray())
acc_test=accuracy_score(test_label, preds11)
acc_train=accuracy_score(train_label, advancedmodel.predict(advancedtrain))

print('SGD 2 accuracy test: ',acc_test )
print('SDG 2 accuracy train: ',acc_train )
show_profit(advancedmodel, 0.5)
class NBSVM(six.with_metaclass(ABCMeta, BaseEstimator, ClassifierMixin)):

    def __init__(self, alpha=1.0, C=1.0, max_iter=10000):
        self.alpha = alpha
        self.max_iter = max_iter
        self.C = C
        self.svm_ = [] # fuggly

    def fit(self, X, y):
        X, y = check_X_y(X, y, 'csr')
        _, n_features = X.shape

        labelbin = LabelBinarizer()
        Y = labelbin.fit_transform(y)
        self.classes_ = labelbin.classes_
        if Y.shape[1] == 1:
            Y = np.concatenate((1 - Y, Y), axis=1)

        # LabelBinarizer().fit_transform() returns arrays with dtype=np.int64.
        # so we don't have to cast X to floating point
        Y = Y.astype(np.float64)

        # Count raw events from data
        n_effective_classes = Y.shape[1]
        self.class_count_ = np.zeros(n_effective_classes, dtype=np.float64)
        self.ratios_ = np.full((n_effective_classes, n_features), self.alpha,
                                 dtype=np.float64)
        self._compute_ratios(X, Y)

        # flugglyness
        for i in range(n_effective_classes):
            X_i = X.multiply(self.ratios_[i])
            svm = LinearSVC(C=self.C, max_iter=self.max_iter)
            Y_i = Y[:,i]
            svm.fit(X_i, Y_i)
            self.svm_.append(svm) 

        return self

    def predict(self, X):
        n_effective_classes = self.class_count_.shape[0]
        n_examples = X.shape[0]

        D = np.zeros((n_effective_classes, n_examples))

        for i in range(n_effective_classes):
            X_i = X.multiply(self.ratios_[i])
            D[i] = self.svm_[i].decision_function(X_i)
        
        return self.classes_[np.argmax(D, axis=0)]
        
    def _compute_ratios(self, X, Y):
        """Count feature occurrences and compute ratios."""
        if np.any((X.data if issparse(X) else X) < 0):
            raise ValueError("Input X must be non-negative")

        self.ratios_ += safe_sparse_dot(Y.T, X)  # ratio + feature_occurrance_c
        normalize(self.ratios_, norm='l1', axis=1, copy=False)
        row_calc = lambda r: np.log(np.divide(r, (1 - r)))
        self.ratios_ = np.apply_along_axis(row_calc, axis=1, arr=self.ratios_)
        check_array(self.ratios_)
        self.ratios_ = sparse.csr_matrix(self.ratios_)

        #p_c /= np.linalg.norm(p_c, ord=1)
        #ratios[c] = np.log(p_c / (1 - p_c))


def f1_class(pred, truth, class_val):
    n = len(truth)

    truth_class = 0
    pred_class = 0
    tp = 0

    for ii in range(0, n):
        if truth[ii] == class_val:
            truth_class += 1
            if truth[ii] == pred[ii]:
                tp += 1
                pred_class += 1
                continue;
        if pred[ii] == class_val:
            pred_class += 1

    precision = tp / float(pred_class)
    recall = tp / float(truth_class)

    return (2.0 * precision * recall) / (precision + recall)


def semeval_senti_f1(pred, truth, pos=2, neg=0): 

    f1_pos = f1_class(pred, truth, pos)
    f1_neg = f1_class(pred, truth, neg)

    return (f1_pos + f1_neg) / 2.0;


def main(train_file, test_file, ngram=(1, 3)):
    print('loading...')
    train = pd.read_csv(train_file, delimiter='\t', encoding='utf-8', header=0,
                        names=['text', 'label'])

    # to shuffle:
    #train.iloc[np.random.permutation(len(df))]

    test = pd.read_csv(test_file, delimiter='\t', encoding='utf-8', header=0,
                        names=['text', 'label'])

    print('vectorizing...')
    vect = CountVectorizer()
    classifier = NBSVM()

    # create pipeline
    clf = Pipeline([('vect', vect), ('nbsvm', classifier)])
    params = {
        'vect__token_pattern': r"\S+",
        'vect__ngram_range': ngram, 
        'vect__binary': True
    }
    clf.set_params(**params)

    #X_train = vect.fit_transform(train['text'])
    #X_test = vect.transform(test['text'])

    print('fitting...')
    clf.fit(train['text'], train['label'])

    print('classifying...')
    pred = clf.predict(test['text'])
   
    print('testing...')
    acc = accuracy_score(test['label'], pred)
    f1 = semeval_senti_f1(pred, test['label'])
    print('NBSVM: acc=%f, f1=%f' % (acc, f1))
advancedvectorizer = TfidfVectorizer( min_df=0.1, max_df=0.8, max_features = 200000, ngram_range = (1, 1))
advancedtrain = advancedvectorizer.fit_transform(trainheadlines)
print(advancedtrain.shape)
advancedmodel = NBSVM(C=0.01)
advancedmodel = advancedmodel.fit(advancedtrain, train_label)
testheadlines = []
for row in range(0,len(test.index)):
    testheadlines.append(' '.join(str(x) for x in test.iloc[row,2:27]))
advancedtest = advancedvectorizer.transform(testheadlines)
preds12 = advancedmodel.predict(advancedtest)
acc_test=accuracy_score(test_label, preds12)
acc_train=accuracy_score(train_label, advancedmodel.predict(advancedtrain))

print('NBSVM 2 accuracy test: ',acc_test )
print('NBSVM 2 accuracy train: ',acc_train )
advancedvectorizer = TfidfVectorizer( min_df=0.031, max_df=0.2, max_features = 200000, ngram_range = (2, 2))
advancedtrain = advancedvectorizer.fit_transform(trainheadlines)
print(advancedtrain.shape)
advancedmodel = NBSVM(C=0.01)
advancedmodel = advancedmodel.fit(advancedtrain, train_label)
testheadlines = []
for row in range(0,len(test.index)):
    testheadlines.append(' '.join(str(x) for x in test.iloc[row,2:27]))
advancedtest = advancedvectorizer.transform(testheadlines)
preds13 = advancedmodel.predict(advancedtest)
acc_test=accuracy_score(test_label, preds13)
acc_train=accuracy_score(train_label, advancedmodel.predict(advancedtrain))

print('NBSVM 2 accuracy test: ',acc_test )
print('NBSVM 2 accuracy train: ',acc_train )
print('NBSVM 2: ', acc13)
batch_size = 32
nb_classes = 2
# advancedvectorizer = TfidfVectorizer( min_df=0.04, max_df=0.5, max_features = 200000, ngram_range = (2, 2))
advancedvectorizer = TfidfVectorizer( min_df=0.01, max_df=0.5, max_features = 200000, ngram_range = (1, 2))
advancedtrain = advancedvectorizer.fit_transform(trainheadlines)
testheadlines = []
for row in range(0,len(test.index)):
    testheadlines.append(clean_text(' '.join(str(x) for x in test.iloc[row,1:26])))
advancedtest = advancedvectorizer.transform(testheadlines)
print(advancedtrain.shape)
X_train = advancedtrain.toarray()
X_test = advancedtest.toarray()

print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)
y_train = np.array(train_label)
y_test = np.array(test_label)

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)


# pre-processing: divide by max and substract mean
scale = np.max(X_train)
X_train /= scale
X_test /= scale

mean = np.mean(X_train)
X_train -= mean
X_test -= mean

input_dim = X_train.shape[1]

# Here's a Deep Dumb MLP (DDMLP)
model = Sequential()
model.add(Dense(256, input_dim=input_dim))
model.add(Activation('relu'))
model.add(Dropout(0.4))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.4))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

# we'll use categorical xent for the loss, and RMSprop as the optimizer
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

print("Training...")
model.fit(X_train, Y_train, epochs=1, batch_size=16, validation_split=0.05)

print("Generating test predictions...")
preds14 = model.predict_classes(X_test, verbose=0)
acc14 = accuracy_score(test_label, preds14)

print('prediction accuracy: ', acc14)
preds14_proba = model.predict_proba(X_test)
t = 0.55

preds14 = (preds14_proba[:, 1] > t) * 1 + (preds14_proba[:, 1] < (1 - t)) * 0 + (preds14_proba[:, 1] >= (1-t)) * (preds14_proba[:, 1] <= t) * -1

test_merged = test.merge(djia, on='Date')
test_merged['preds'] = preds14

test_merged['Open_shift'] = test_merged['Open'].shift(1)
test_merged = test_merged.dropna()
test_merged['profit'] = (np.abs((test_merged['Open_shift'] - test_merged['Open'])) * (test_merged['Label_adj'] == test_merged['preds']) - np.abs((test_merged['Open_shift'] - test_merged['Open'])) * (test_merged['Label_adj'] != test_merged['preds'])) * (test_merged['preds'] != -1)
# test_merged = test_merged.sort_values(by='Date', ascending=True)
test_merged['cum_profit'] = test_merged['profit'].cumsum()

plt.plot(test_merged['cum_profit'])
plt.show()
max_features = 10000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.1
maxlen = 500
batch_size = 32
nb_classes = 2
# vectorize the text samples into a 2D integer tensor
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(trainheadlines)
sequences_train = tokenizer.texts_to_sequences(trainheadlines)
sequences_test = tokenizer.texts_to_sequences(testheadlines)
print('Pad sequences (samples x time)')
X_train = sequence.pad_sequences(sequences_train, maxlen=maxlen)
X_test = sequence.pad_sequences(sequences_test, maxlen=maxlen)

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)


print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)
tokenizer.word_index
Y_train
pd.Series(X_test.ravel()).value_counts()
X_test
basicmodel = LogisticRegression(C=1.0)
basicmodel = basicmodel.fit(X_train, Y_train[:, 0])

preds1 = basicmodel.predict(X_test)
acc_test=accuracy_score(Y_test[:, 0], preds1)
acc_train=accuracy_score(Y_train[:, 0], basicmodel.predict(X_train))

print('Logic Regression 1 accuracy test: ',acc_test )
print('Logic Regression 1 accuracy train: ',acc_train )
basicmodel.coef_
print('Build model...')
model = Sequential()
model.add(Embedding(max_features, 128))
model.add(LSTM(128, dropout=0.5)) 
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')
model.fit(X_train, Y_train, batch_size=batch_size, epochs=2) #,
         #validation_data=(X_test, Y_test))
score, acc = model.evaluate(X_test, Y_test,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)


print("Generating test predictions...")
preds15 = model.predict_classes(X_test, verbose=0)
acc15 = accuracy_score(test_label, preds15)
preds15_proba = model.predict_proba(X_test)
t = 0.55

preds15 = (preds15_proba[:, 1] > t) * 1 + (preds15_proba[:, 1] < (1 - t)) * 0 + (preds15_proba[:, 1] >= (1-t)) * (preds15_proba[:, 1] <= t) * -1

test_merged = test.merge(djia, on='Date')
test_merged['preds'] = preds15

test_merged['Open_shift'] = test_merged['Open'].shift(1)
test_merged = test_merged.dropna()
test_merged['profit'] = (np.abs((test_merged['Open_shift'] - test_merged['Open'])) * (test_merged['Label_adj'] == test_merged['preds']) - np.abs((test_merged['Open_shift'] - test_merged['Open'])) * (test_merged['Label_adj'] != test_merged['preds'])) * (test_merged['preds'] != -1)
# test_merged = test_merged.sort_values(by='Date', ascending=True)
test_merged['cum_profit'] = test_merged['profit'].cumsum()

plt.plot(test_merged['cum_profit'])
plt.show()
nb_filter = 120
filter_length = 2
hidden_dims = 120
nb_epoch = 2
print('Build model...')
model = Sequential()
model.add(Embedding(max_features, 128, dropout=0.2))
# we add a Convolution1D, which will learn nb_filter
# word group filters of size filter_length:
model.add(Convolution1D(nb_filter=nb_filter,
                        filter_length=filter_length,
                        border_mode='valid',
                        activation='relu',
                        subsample_length=1))

def max_1d(X):
    return K.max(X, axis=1)

model.add(Lambda(max_1d, output_shape=(nb_filter,)))
model.add(Dense(hidden_dims)) 
model.add(Dropout(0.2)) 
model.add(Activation('relu'))
model.add(Dense(nb_classes))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
print('Train...')
model.fit(X_train, Y_train, batch_size=32, nb_epoch=1,
          validation_data=(X_test, Y_test))
score, acc = model.evaluate(X_test, Y_test,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)


print("Generating test predictions...")
preds16 = model.predict_classes(X_test, verbose=0)
acc16 = accuracy_score(test['Label'], preds16)
print('prediction accuracy: ', acc16)



