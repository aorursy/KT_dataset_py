import re
import string
import numpy as np 
import random
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from plotly import graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff
from collections import Counter

from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator


import nltk
from nltk.corpus import stopwords

from tqdm import tqdm, tqdm_notebook
import os
import nltk
import spacy
import random
from spacy.util import compounding
from spacy.util import minibatch

import warnings
warnings.filterwarnings("ignore")

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold,  cross_val_score

# Libraries
from scipy import stats
#from scipy.sparse import hstack, csr_matrix
#from sklearn.model_selection import train_test_split, KFold

#import xgboost as xgb
#from sklearn import model_selection
from sklearn.metrics import accuracy_score
import json
# import ast
# import eli5
# import shap
#from catboost import CatBoostRegressor
from urllib.request import urlopen
# from PIL import Image
#from sklearn.preprocessing import LabelEncoder
import time
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn import linear_model

from hyperopt import hp, tpe
from hyperopt.fmin import fmin

#from sklearn.model_selection import
#from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer
# import lightgbm as lgbm
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble

from keras.preprocessing import text, sequence
import xgboost
import scipy
import shap
plt.style.use('fivethirtyeight')
df_train = pd.read_csv('../input/tweet-sentiment-extraction/train.csv')
df_test = pd.read_csv('../input/tweet-sentiment-extraction/test.csv')
df_submission = pd.read_csv('../input/tweet-sentiment-extraction/sample_submission.csv')
df_train.head()
df_test = df_test.assign(selected_text=lambda x: x.text)
#df_all.shape
df_train[df_train['selected_text'].isna()]
df_train[df_train['text'].isna()]
df_train.iloc[314]['text'] = 'None'
df_train.iloc[314]['selected_text'] = 'None'
df_all = pd.concat([df_train, df_test])
df_all[df_all.text.isna()]
df_train.sentiment.unique()
# encode the target variable
# label encode the target variable 
encoder = preprocessing.LabelEncoder()
train_y_POS = encoder.fit_transform(df_train.sentiment=='positive')
test_y_POS = encoder.fit_transform(df_test.sentiment=='positive')
np.where(train_y_POS==1)
np.where(df_train.sentiment=='positive')
train_y_NEG = encoder.fit_transform(df_train.sentiment=='negative')
test_y_NEG = encoder.fit_transform(df_test.sentiment=='negative')
from nltk.tokenize import TweetTokenizer
# create a count vectorizer object 
tweet_token = TweetTokenizer(preserve_case=True, strip_handles=True)
#list(df_train['text'].fillna(""))
df_train['text'].apply(lambda x: tweet_token.tokenize(x))
def tweet_token_proces(x):
    return tweet_token.tokenize(x)
count_vect = CountVectorizer(tokenizer=tweet_token_proces)
count_vect.fit(df_all['text']) ## fillna
column_index = count_vect.get_feature_names()
# transform the training and validation data using count vectorizer object
xtrain_count =  count_vect.transform(df_train.text)
xtest_count =  count_vect.transform(df_test.text)
# ## tf idf

# # word level tf-idf
# tfidf_vect = TfidfVectorizer(tokenizer=tweet_token_proces, max_features=5000)
# tfidf_vect.fit(df_all['text'].fillna(' '))
# xtrain_tfidf =  tfidf_vect.transform(df_train.text.fillna(' '))
# xtest_tfidf =  tfidf_vect.transform(df_test.text.fillna(' '))
# # ngram level tf-idf 
# tfidf_vect_ngram = TfidfVectorizer(tokenizer=tweet_token_proces, ngram_range=(2,5), max_features=5000)
# tfidf_vect_ngram.fit(df_all['text'].fillna(' '))
# xtrain_tfidf_ngram =  tfidf_vect_ngram.transform(df_train.text.fillna(' '))
# xtest_tfidf_ngram =  tfidf_vect_ngram.transform(df_test.text.fillna(' '))

# # characters level tf-idf
# tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2,5), max_features=5000)
# tfidf_vect_ngram_chars.fit(df_all['text'].fillna(' '))
# xtrain_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(df_train.text.fillna(' ')) 
# xtest_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(df_test.text.fillna(' ')) 
# # load the pre-trained word-embedding vectors 
# embeddings_index = {}
# for i, line in enumerate(open('../input/wiki-news-300d-1M.vec')):
#     values = line.split()
#     embeddings_index[values[0]] = np.asarray(values[1:], dtype='float32')
# # create a tokenizer 
# # use keras.preprocessing.text
# token = text.Tokenizer()
# token.fit_on_texts(df_all['text'].fillna(' '))
# word_index = token.word_index
# # convert text to sequence of tokens and pad them to ensure equal length vectors 
# train_seq_x = sequence.pad_sequences(token.texts_to_sequences(df_train.text.fillna(' ')), maxlen=70)
# test_seq_x = sequence.pad_sequences(token.texts_to_sequences(df_test.text.fillna(' ')), maxlen=70)
# # create token-embedding mapping
# embedding_matrix = np.zeros((len(word_index) + 1, 300))
# for word, i in word_index.items():
#     embedding_vector = embeddings_index.get(word)
#     if embedding_vector is not None:
#         embedding_matrix[i] = embedding_vector
# df_train['text'] = df_train['text'].fillna('None')
# df_test['text'] = df_test['text'].fillna('None')
# df_train[df_train.text=='None']
# df_train['char_count'] = df_train['text'].apply(len)
# df_test['char_count'] = df_test['text'].apply(len)

# df_train['word_count'] = df_train['text'].apply(lambda x: len(x.split()))
# df_test['word_count'] = df_test['text'].apply(lambda x: len(x.split()))

# df_train['word_density'] = df_train['char_count'] / (df_train['word_count']+1)
# df_test['word_density'] = df_test['char_count'] / (df_test['word_count']+1)

# df_train['punctuation_count'] = df_train['text'].apply(lambda x: len("".join(_ for _ in x if _ in string.punctuation)))
# df_test['punctuation_count'] = df_test['text'].apply(lambda x: len("".join(_ for _ in x if _ in string.punctuation))) 

# df_train['title_word_count'] = df_train['text'].apply(lambda x: len([wrd for wrd in x.split() if wrd.istitle()]))
# df_test['title_word_count'] = df_test['text'].apply(lambda x: len([wrd for wrd in x.split() if wrd.istitle()]))

# df_train['upper_case_word_count'] = df_train['text'].apply(lambda x: len([wrd for wrd in x.split() if wrd.isupper()]))
# df_test['upper_case_word_count'] = df_test['text'].apply(lambda x: len([wrd for wrd in x.split() if wrd.isupper()]))
## from the original notebook

def jaccard(str1, str2): 
    a = set(str1.lower().split()) 
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))

results_jaccard=[]

for ind,row in df_train.iterrows():
    sentence1 = row.text
    sentence2 = row.selected_text
    #print(ind)
    jaccard_score = jaccard(sentence1,sentence2)
    results_jaccard.append(jaccard_score)

df_train['jaccard'] = results_jaccard

df_train['Num_words_ST'] = df_train['selected_text'].apply(lambda x:len(str(x).split())) #Number Of words in Selected Text
df_train['Num_word_text'] = df_train['text'].apply(lambda x:len(str(x).split())) #Number Of words in main text
df_train['difference_in_words'] = df_train['Num_word_text'] - df_train['Num_words_ST'] #Difference in Number of words text and Selected Text

results_jaccard=[]

for ind,row in df_test.iterrows():
    sentence1 = row.text
    sentence2 = row.selected_text

    jaccard_score = jaccard(sentence1,sentence2)
    results_jaccard.append(jaccard_score)

df_test['jaccard'] = results_jaccard
df_test['Num_words_ST'] = df_test['selected_text'].apply(lambda x:len(str(x).split())) #Number Of words in Selected Text
df_test['Num_word_text'] = df_test['text'].apply(lambda x:len(str(x).split())) #Number Of words in main text
df_test['difference_in_words'] = df_test['Num_word_text'] - df_test['Num_words_ST'] #Difference in Number of words text and Selected Text
df_train.iloc[314]
def train_model(classifier, feature_vector_train, label, feature_vector_test,  test_y,is_neural_net=False):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)
    
    # predict the labels on testation dataset
    predictions = classifier.predict(feature_vector_test)
    
    if is_neural_net:
        predictions = predictions.argmax(axis=-1)
    
    #return metrics.accuracy_score(predictions, test_y)
    return metrics.accuracy_score(predictions, test_y)

# Extereme Gradient Boosting on Count Vectors
## Positive
accuracy = train_model(xgboost.XGBClassifier(), xtrain_count.tocsc(), train_y_POS, xtest_count.tocsc(), test_y_POS)
print("Xgb, Count Vectors POS: ", accuracy)
## negative 
accuracy = train_model(xgboost.XGBClassifier(), xtrain_count.tocsc(), train_y_NEG, xtest_count.tocsc(), test_y_NEG)
print("Xgb, Count Vectors NEG: ", accuracy)

# # Extereme Gradient Boosting on Word Level TF IDF Vectors
# ## POS
# accuracy = train_model(xgboost.XGBClassifier(), xtrain_tfidf.tocsc(), train_y_POS, xtest_tfidf.tocsc(), test_y_POS)
# print("Xgb, WordLevel TF-IDF POS: ", accuracy)
# ## NEG
# accuracy = train_model(xgboost.XGBClassifier(), xtrain_tfidf.tocsc(), train_y_NEG, xtest_tfidf.tocsc(), test_y_NEG)
# print("Xgb, WordLevel TF-IDF NEG: ", accuracy)
# # Extereme Gradient Boosting on ngram TF IDF Vectors
# ## POS
# accuracy = train_model(xgboost.XGBClassifier(), xtrain_tfidf_ngram.tocsc(), train_y_POS, \
#                        xtest_tfidf_ngram.tocsc(), test_y_POS)
# print("Xgb, ngram Vectors POS: ", accuracy)

# ## NEG
# accuracy = train_model(xgboost.XGBClassifier(), xtrain_tfidf_ngram.tocsc(), train_y_NEG, \
#                        xtest_tfidf_ngram.tocsc(), test_y_NEG)
# print("Xgb, ngram Vectors NEG: ", accuracy)
# # Extereme Gradient Boosting on char level tfidf
# ## POS
# accuracy = train_model(xgboost.XGBClassifier(), xtrain_tfidf_ngram_chars.tocsc(), train_y_POS,\
#                        xtest_tfidf_ngram_chars.tocsc(), test_y_POS)
# print("Xgb, CharLevel Vectors POS: ", accuracy)

# ## NEG
# accuracy = train_model(xgboost.XGBClassifier(), xtrain_tfidf_ngram_chars.tocsc(), train_y_NEG, \
#                        xtest_tfidf_ngram_chars.tocsc(), test_y_NEG)
# print("Xgb, CharLevel Vectors NEG: ", accuracy)
#type(train_seq_x)
# # Extereme Gradient Boosting on word embeddings
# ## POS
# accuracy = train_model(xgboost.XGBClassifier(), train_seq_x, train_y_POS,\
#                        test_seq_x, test_y_POS)
# print("Xgb, word embedding POS: ", accuracy)

# ## NEG
# accuracy = train_model(xgboost.XGBClassifier(), train_seq_x, train_y_NEG, \
#                        test_seq_x, test_y_NEG)
# print("Xgb, word embedding NEG: ", accuracy)
# ######### tfidf and word cout
# xtrain_tfidf_wc = scipy.sparse.hstack((xtrain_tfidf, xtrain_count))
# xtest_tfidf_wc = scipy.sparse.hstack((xtest_tfidf, xtest_count))
# ## POS
# accuracy = train_model(xgboost.XGBClassifier(), xtrain_tfidf_wc.tocsc(), train_y_POS, \
#                       xtest_tfidf_wc.tocsc(), test_y_POS)

# print("POS: ", accuracy)

# ## NEG
# accuracy = train_model(xgboost.XGBClassifier(), xtrain_tfidf_wc.tocsc(), train_y_NEG, \
#                       xtest_tfidf_wc.tocsc(), test_y_NEG)

# print("NEG: ", accuracy)
# ########### all tfidf features

# xtrain_tfidf_all = scipy.sparse.hstack((xtrain_tfidf_ngram_chars, xtrain_tfidf, xtrain_tfidf_ngram))

# xtest_tfidf_all = scipy.sparse.hstack((xtest_tfidf_ngram_chars, xtest_tfidf, xtest_tfidf_ngram))


# ## POS
# accuracy = train_model(xgboost.XGBClassifier(), xtrain_tfidf_all.tocsc(), train_y_POS, \
#                       xtest_tfidf_all.tocsc(), test_y_POS)

# print("POS: ", accuracy)

# ## NEG
# accuracy = train_model(xgboost.XGBClassifier(), xtrain_tfidf_all.tocsc(), train_y_NEG, \
#                       xtest_tfidf_all.tocsc(), test_y_NEG)

# print("NEG: ", accuracy)
# ###########  tfidf features with count

# xtrain_tfidf_all_2 = scipy.sparse.hstack(( xtrain_tfidf, xtrain_tfidf_ngram, xtrain_count))

# xtest_tfidf_all_2 = scipy.sparse.hstack(( xtest_tfidf, xtest_tfidf_ngram, xtest_count))


# ## POS
# accuracy = train_model(xgboost.XGBClassifier(), xtrain_tfidf_all_2.tocsc(), train_y_POS, \
#                       xtest_tfidf_all_2.tocsc(), test_y_POS)

# print("POS: ", accuracy)

# ## NEG
# accuracy = train_model(xgboost.XGBClassifier(), xtrain_tfidf_all_2.tocsc(), train_y_NEG, \
#                       xtest_tfidf_all_2.tocsc(), test_y_NEG)

# print("NEG: ", accuracy)
# ###########  tfidf features with count

# xtrain_tfidf_all_3 = scipy.sparse.hstack(( xtrain_tfidf, xtrain_tfidf_ngram))

# xtest_tfidf_all_3 = scipy.sparse.hstack(( xtest_tfidf, xtest_tfidf_ngram))


# ## POS
# accuracy = train_model(xgboost.XGBClassifier(), xtrain_tfidf_all_3.tocsc(), train_y_POS, \
#                       xtest_tfidf_all_3.tocsc(), test_y_POS)

# print("POS: ", accuracy)

# ## NEG
# accuracy = train_model(xgboost.XGBClassifier(), xtrain_tfidf_all_3.tocsc(), train_y_NEG, \
#                       xtest_tfidf_all_3.tocsc(), test_y_NEG)

# print("NEG: ", accuracy)
# ########### tfidf nlp wc
# nlp_features = ['char_count', 'word_count', 'word_density', \
#                 'punctuation_count', 'title_word_count','upper_case_word_count']
# xtrain_tfidf_wc_nlp = scipy.sparse.hstack((xtrain_tfidf, xtrain_count, df_train[nlp_features]))
# xtest_tfidf_wc_nlp = scipy.sparse.hstack((xtest_tfidf, xtest_count, df_test[nlp_features]))
# ## POS
# accuracy = train_model(xgboost.XGBClassifier(), xtrain_tfidf_wc_nlp.tocsc(), train_y_POS, \
#                       xtest_tfidf_wc_nlp.tocsc(), test_y_POS)

# print("POS: ", accuracy)

# ## NEG
# accuracy = train_model(xgboost.XGBClassifier(), xtrain_tfidf_wc_nlp.tocsc(), train_y_NEG, \
#                       xtest_tfidf_wc_nlp.tocsc(), test_y_NEG)

# print("NEG: ", accuracy)

######### tfidf wc special

# xtrain_tfidf_wc_special = scipy.sparse.hstack((xtrain_tfidf, xtrain_count, df_train[special_features]))
# xtest_tfidf_wc_special = scipy.sparse.hstack((xtest_tfidf, xtest_count, df_test[special_features]))
# ## POS
# accuracy = train_model(xgboost.XGBClassifier(), xtrain_tfidf_wc_special.tocsc(), train_y_POS, \
#                       xtest_tfidf_wc_special.tocsc(), test_y_POS)

# print("POS: ", accuracy)

# ## NEG
# accuracy = train_model(xgboost.XGBClassifier(), xtrain_tfidf_wc_special.tocsc(), train_y_NEG, \
#                       xtest_tfidf_wc_special.tocsc(), test_y_NEG)

# print("NEG: ", accuracy)


########## mash up all features together

# xtrain_all = scipy.sparse.hstack((xtrain_tfidf, xtrain_count, df_train[special_features + nlp_features]))
# xtest_all = scipy.sparse.hstack((xtest_tfidf, xtest_count, df_test[special_features + nlp_features]))

# ## POS
# accuracy = train_model(xgboost.XGBClassifier(), xtrain_all.tocsc(), train_y_POS, \
#                       xtest_all.tocsc(), test_y_POS)

# print("POS: ", accuracy)

# ## NEG
# accuracy = train_model(xgboost.XGBClassifier(), xtrain_all.tocsc(), train_y_NEG, \
#                       xtest_all.tocsc(), test_y_NEG)

# print("NEG: ", accuracy)



baseline = xgboost.XGBClassifier()
model_baseline = baseline.fit(X=xtrain_count, y=train_y_POS)
model_baseline.get_params()
def objective(params):
    params = {
        'max_depth': int(params['max_depth']),
        'gamma': "{:.3f}".format(params['gamma']),
        'subsample': "{:.2f}".format(params['subsample']),
        'reg_alpha': "{:.3f}".format(params['reg_alpha']),
        'reg_lambda': "{:.3f}".format(params['reg_lambda']),
        'learning_rate': "{:.3f}".format(params['learning_rate']),
        'num_leaves': '{:.3f}'.format(params['num_leaves']),
        'colsample_bytree': '{:.3f}'.format(params['colsample_bytree']),
        'min_child_samples': '{:.3f}'.format(params['min_child_samples']),
        'feature_fraction': '{:.3f}'.format(params['feature_fraction']),
        'bagging_fraction': '{:.3f}'.format(params['bagging_fraction'])
    }
    
    clf = xgboost.XGBClassifier(
        n_estimators=100,
        #learning_rate=0.1,
        n_jobs=4,
        
        **params
    )
    
    score = cross_val_score(clf, xtrain_count, train_y_POS, scoring='accuracy', cv=KFold(n_splits=5)).mean()
    print("accuracy {:.3f} params {}".format(score, params))
    return -score
## code here is mostly borrowed from here: https://www.kaggle.com/kabure/extensive-eda-and-modeling-xgb-hyperopt
space = {
    # The maximum depth of a tree, same as GBM.
    # Used to control over-fitting as higher depth will allow model 
    # to learn relations very specific to a particular sample.
    # Should be tuned using CV.
    # Typical values: 3-10
    'max_depth': hp.quniform('max_depth', 3, 6, 1),
    
    # reg_alpha: L1 regularization term. L1 regularization encourages sparsity 
    # (meaning pulling weights to 0). It can be more useful when the objective
    # is logistic regression since you might need help with feature selection.
    'reg_alpha':  hp.uniform('reg_alpha', 0.01, 0.4),
    
    # reg_lambda: L2 regularization term. L2 encourages smaller weights, this
    # approach can be more useful in tree-models where zeroing 
    # features might not make much sense.
    'reg_lambda': hp.uniform('reg_lambda', 0.01, .4),
    
    # eta: Analogous to learning rate in GBM
    # Makes the model more robust by shrinking the weights on each step
    # Typical final values to be used: 0.01-0.2
    'learning_rate': hp.uniform('learning_rate', 0.01, 0.2),
    
    # colsample_bytree: Similar to max_features in GBM. Denotes the 
    # fraction of columns to be randomly samples for each tree.
    # Typical values: 0.5-1
    'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
    
    # A node is split only when the resulting split gives a positive
    # reduction in the loss function. Gamma specifies the 
    # minimum loss reduction required to make a split.
    # Makes the algorithm conservative. The values can vary depending on the loss function and should be tuned.
    'gamma': hp.uniform('gamma', 0.01, .7),
    
    # more increases accuracy, but may lead to overfitting.
    # num_leaves: the number of leaf nodes to use. Having a large number 
    # of leaves will improve accuracy, but will also lead to overfitting.
    'num_leaves': hp.choice('num_leaves', list(range(20, 250, 10))),
    
    # specifies the minimum samples per leaf node.
    # the minimum number of samples (data) to group into a leaf. 
    # The parameter can greatly assist with overfitting: larger sample
    # sizes per leaf will reduce overfitting (but may lead to under-fitting).
    'min_child_samples': hp.choice('min_child_samples', list(range(100, 250, 10))),
    
    # subsample: represents a fraction of the rows (observations) to be 
    # considered when building each subtree. Tianqi Chen and Carlos Guestrin
    # in their paper A Scalable Tree Boosting System recommend 
    'subsample': hp.choice('subsample', [0.2, 0.4, 0.5, 0.6, 0.7, .8, .9]),
    
    # randomly select a fraction of the features.
    # feature_fraction: controls the subsampling of features used
    # for training (as opposed to subsampling the actual training data in 
    # the case of bagging). Smaller fractions reduce overfitting.
    'feature_fraction': hp.uniform('feature_fraction', 0.8, .9),
    
    # randomly bag or subsample training data.
    'bagging_fraction': hp.uniform('bagging_fraction', 0.8, .9)
    
    # bagging_fraction and bagging_freq: enables bagging (subsampling) 
    # of the training data. Both values need to be set for bagging to be used.
    # The frequency controls how often (iteration) bagging is used. Smaller
    # fractions and frequencies reduce overfitting.
}
%%time
# this steps takes at least 20mins
# best = fmin(fn=objective,
#             space=space,
#             algo=tpe.suggest, max_evals=30) 
best = {'bagging_fraction': 0.8768575337571937,
 'colsample_bytree': 0.9933592930641432,
 'feature_fraction': 0.816825176108506,
 'gamma': 0.05587328363633812,
 'learning_rate': 0.19879098664834996,
 'max_depth': 6.0,
 'min_child_samples': 9,
 'num_leaves': 7,
 'reg_alpha': 0.11806338517600543,
 'reg_lambda': 0.23269341544465222,
 'subsample': 0.6}
best['max_depth'] = 6
best['subsample'] = .6
#params = {'max_depth': 5, 'gamma': '0.332', 'subsample': '0.80', 'reg_alpha': '0.365', 'reg_lambda': '0.070', 'learning_rate': '0.200', 'num_leaves': '150.000', 'colsample_bytree': '0.792', 'min_child_samples': '120.000', 'feature_fraction': '0.710', 'bagging_fraction': '0.436'}
params = best
#params['max_depth'] = 6
#params['subsample'] = .9
xgb = xgboost.XGBClassifier(**params, n_estimators=100,
        #learning_rate=0.05,
        n_jobs=4)
## POSITIVE
xgb_fit_POS = xgb.fit(xtrain_count.tocsc(), train_y_POS)

predictions = xgb_fit_POS.predict(xtest_count.tocsc())

metrics.accuracy_score(predictions, test_y_POS) ## improved from 0.816
## NEGATIVE
xgb_neg = xgboost.XGBClassifier(**params, n_estimators=100,
        #learning_rate=0.05,
        n_jobs=4)
xgb_fit_NEG = xgb_neg.fit(xtrain_count.tocsc(), train_y_NEG)
predictions = xgb_fit_NEG.predict(xtest_count.tocsc())
metrics.accuracy_score(predictions, test_y_NEG)
xtrain_all_dense = pd.DataFrame(xtrain_count.tocsc().todense())
xtrain_all_dense.shape
column_index = count_vect.get_feature_names()
len(column_index)
shap.initjs()
#xgb_fit_POS
#import shap
explainer = shap.TreeExplainer(xgb_fit_POS)
shap_values = explainer.shap_values(xtrain_all_dense)

# visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)
shap.force_plot(explainer.expected_value, shap_values[1,:],\
                xtrain_all_dense.iloc[1,:])
column_index[10248], column_index[8282]
np.mean(shap_values[:, 10248]), np.mean(shap_values[:, 8282])
shap_values.shape
shap.force_plot(explainer.expected_value, shap_values[9011,:],\
                xtrain_all_dense.iloc[9011,:])
column_index[4898], column_index[7229]
np.mean(shap_values[:, 4898]), np.mean(shap_values[:, 7229])

def get_word_index(x):
    result = []
    for i in x.split(' '):
        try:
            #current_word = wordnet_lemmatizer.lemmatize(i.lower())
            result.append(column_index.index(i))
        except ValueError:
            pass
            #result.append(-1)
    return result

def get_word(x):
    result = []
    for i in x.split(' '):
        try:
            #current_word = wordnet_lemmatizer.lemmatize(i.lower())
            column_index.index(i)
            result.append(i)
        except ValueError:
            pass
            #result.append('None')
           
    return result
column_index.index('making')
df_train['text_word_index'] = df_train.text.apply(lambda x: get_word_index(x))
df_train['text_word_has_index'] = df_train.text.apply(lambda x: get_word(x))
## now apply the same for selected text
df_train['selected_text_word_index'] = df_train.selected_text.apply(lambda x: get_word_index(x))
df_train['selected_text_word_has_index'] = df_train.selected_text.apply(lambda x: get_word(x))
## now map all the text for each words with shap value
shap_extracted = []


for i in np.arange(len(df_train)):
    shap_extracted.append(shap_values[i, df_train['text_word_index'][i]])
shap_extracted_mean = [np.mean(i) for i in shap_extracted]
## now do the same for selected text
selected_shap_extracted = []
for i in np.arange(len(df_train)):
    selected_shap_extracted.append(shap_values[i, df_train['selected_text_word_index'][i]])
selected_shap_extracted_mean = [np.mean(i) for i in selected_shap_extracted]
df_train['shap_extracted'] = shap_extracted
df_train['shap_extracted_mean'] = shap_extracted_mean
df_train['selected_shap_extracted_mean'] = selected_shap_extracted_mean
sns.stripplot(df_train.shap_extracted_mean,df_train.sentiment, jitter=True)
sns.stripplot(df_train.selected_shap_extracted_mean, df_train.sentiment,jitter=True )
concat_pd = pd.concat([df_train[df_train.sentiment=='positive']['shap_extracted_mean'],\
           df_train[df_train.sentiment=='positive']['selected_shap_extracted_mean']])

concat_pd = pd.DataFrame(concat_pd).reset_index()
concat_pd['text'] = 'text'
concat_pd.columns = ['index', 'mean', 'text']
concat_pd.loc[8582:, 'text'] = 'selected_text'
sns.stripplot(concat_pd['mean'], concat_pd.text)
df_train['mean_diff'] = df_train.selected_shap_extracted_mean - df_train.shap_extracted_mean
#df_train.mean_diff.hist(bins=30)
sns.kdeplot(df_train[df_train['sentiment']=='positive']['mean_diff'], shade=True, color="b")
plt.figure(figsize=(12,6))
p1=sns.kdeplot(df_train[df_train['sentiment']=='positive']['shap_extracted_mean'], shade=True, color="r")\
.set_title('Kernel Distribution of SHAP values (raw text vs selected text)')
p1=sns.kdeplot(df_train[df_train['sentiment']=='positive']['selected_shap_extracted_mean'], shade=True, color="b")
plt.vlines(x=-0.15, ymax=7, ymin=0, linestyles='dashed' )
plt.vlines(x=0.01, ymax=7, ymin=0, linestyles='dashed')

# uncomment it if you want to run it
# def shap_cutoff_2(cutoff1, cutoff2, x):
#     '''
#     input: cutoff for shap value and x is the extracted shap value
#     output: a tuple with indices for text selection
#     '''
#     try: 
#         min_idx_1 = np.where(x == np.min([i for i in x if i < cutoff1]))[0][0]
#         max_idx_1 = np.where(x==np.max([i for i in x if i < cutoff1]))[0][0]
#         min_idx_2 = np.where(x == np.min([i for i in x if i > cutoff2]))[0][0]
#         max_idx_2 = np.where(x==np.max([i for i in x if i > cutoff2]))[0][0]
        
#         return (min_idx_1, max_idx_1, min_idx_2, max_idx_2)
#     except:
#         # if x is empty, then return an impossivle value
#         return (100, 100, 100, 100)
    
# def jaccard(str1, str2): 
#     a = set(str1.lower().split()) 
#     b = set(str2.lower().split())
#     c = a.intersection(b)
#     return float(len(c)) / (len(a) + len(b) - len(c))

# #for c1, c2 in zip(np.linspace(-.5, -0.01, 15), np.linspace(0.01, 0.5, 15)):
#     ## use a cutoff to find word indices 
# for c1 in np.linspace(-.5, -.3, 5):
#     for c2 in np.linspace(0.5, 0.6, 2):
#         df_train['shap_cutoff_2'] = df_train.shap_extracted.apply(lambda x: shap_cutoff_2(c1,c2, x))
#         partial_text=[]
#         # get selected text
#         for i,row in df_train.iterrows():
#             if row.shap_cutoff_2== (100, 100, 100, 100):
#                 partial_text.append(row.text.split(' '))
#             else:
#                 min_idx = np.min(row.shap_cutoff_2)
#                 max_idx = np.max(row.shap_cutoff_2) + 1
#                 partial_text.append(row.text_word_has_index[min_idx: max_idx])

#         soln_sentence = [' '.join(i) for i in partial_text]
#         df_train['selected_soln_1'] = soln_sentence

#         # calculate jaccard
#         results_jaccard=[]

#         for ind,row in df_train.iterrows():
#             sentence1 = row.selected_text
#             sentence2 = row.selected_soln_1
#             #print(ind)
#             jaccard_score = jaccard(sentence1,sentence2)
#             results_jaccard.append(jaccard_score)
#         df_train['results_jaccard'] = results_jaccard

#         print('cutoff: ', c1, ', ', c2)
#         print('mean jaccard: ',df_train[df_train.sentiment=='positive'].results_jaccard.mean())
#         sns.kdeplot(df_train[df_train['sentiment']=='positive']['results_jaccard'], shade=True, color="b")
#         plt.show()

import numpy as np
from scipy.spatial import distance
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
import lime
import lime.lime_tabular


df_train[df_train.sentiment=='positive'][:10].index
count_vect = CountVectorizer(tokenizer=tweet_token_proces)
grad_boost = GradientBoostingClassifier(max_depth=6)
pipe = make_pipeline(count_vect, grad_boost)
pipe.fit(df_train.text, train_y_POS)
pipe.score(df_test.text, test_y_POS)


from lime.lime_text import LimeTextExplainer
explainer = LimeTextExplainer(class_names=['nonPOS', 'POS'])
idx = [6, 9, 11, 21, 25, 28, 30, 31, 33, 39, 27474]
for i in idx:
    exp = explainer.explain_instance(df_train.text[i], pipe.predict_proba)
    print(df_train.selected_text.values[i])
    exp.show_in_notebook()
    
exp.as_list()
column_index = count_vect.get_feature_names()
# threshould = .07
# select_text_list = []
# for i in tqdm_notebook(range(len(df_train))):
#     row = df_train.iloc[i]
#     if row.sentiment=='positive':
#         try:
#             exp = explainer.explain_instance(row.text, pipe.predict_proba).as_list()
#             exp_words =[k[0] for k in exp if k[1] > threshould]
#             text_list = tweet_token_proces(row.text)
#             idx = [text_list.index(k) for k in exp_words]
#             if len(idx) < 1:
#                 select_text_list.append([])
#             if len(idx) ==1:
#                 select_text_list.append(exp_words[0])
#             else:

#                 max_idx = max(idx)
#                 min_idx = min(idx)
#                 cur_text = text_list[min_idx:max_idx]
#                     #cur_text_processed = [i if i.isalpha() else i + ' ' for i in cur_text]
#                     #print(' '.join(cur_text))
#                 select_text_list.append(cur_text)
#         except:
#             select_text_list.append([])
#     else:
#         select_text_list.append([])



