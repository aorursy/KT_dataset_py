# import required packages



import pandas as pd 

import numpy as np

import os, gc, time, warnings



from scipy.misc import imread

from scipy import sparse

import scipy.stats as ss

from scipy.sparse import csr_matrix, hstack, vstack



import matplotlib.pyplot as plt, matplotlib.gridspec as gridspec 

import seaborn as sns

from wordcloud import WordCloud ,STOPWORDS

from PIL import Image

import matplotlib_venn as venn

import pydot, graphviz

from IPython.display import Image



import string, re, nltk, collections

from nltk.util import ngrams

from nltk.corpus import stopwords

import spacy

from nltk import pos_tag

from nltk.stem import PorterStemmer

from nltk.stem.wordnet import WordNetLemmatizer 

from nltk.tokenize import word_tokenize

from nltk.tokenize import TweetTokenizer   



from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer

from sklearn.decomposition import TruncatedSVD

from sklearn.base import BaseEstimator, ClassifierMixin

from sklearn.utils.validation import check_X_y, check_is_fitted

from sklearn.linear_model import LogisticRegression, SGDClassifier

from sklearn.svm import SVC

from sklearn import metrics

from sklearn.feature_selection import SelectFromModel

from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import train_test_split



import tensorflow as tf

import keras.backend as K

from keras.models import Model, Sequential

from keras.utils import plot_model

from keras.layers import Input, Dense, Embedding, SpatialDropout1D, concatenate, BatchNormalization

from keras.layers import GRU, LSTM, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D, Conv1D

from keras.preprocessing import text, sequence

from keras.callbacks import Callback
# settings



os.environ['OMP_NUM_THREADS'] = '4'

start_time = time.time()

color = sns.color_palette()

sns.set_style("dark")

warnings.filterwarnings("ignore")



eng_stopwords = set(stopwords.words("english"))

lem = WordNetLemmatizer()

ps = PorterStemmer()

tokenizer = TweetTokenizer()



%matplotlib inline
# print the names of files available in the root directory

print(os.listdir('../input'))
# import the dataset



train = pd.read_csv('../input/nslkdd-dataset/KDDTrain.csv')

test = pd.read_csv('../input/nslkdd-dataset/KDDTest.csv')
print("Training data information...")

train.info()
print('Test data information...')

test.info()
# obtaining a new target variable for each attack class



attack_classes = ['back', 'buffer_overflow', 'ftp_write', 'guess_passwd', 'imap', 'ipsweep', 'land', 

                  'loadmodule', 'multihop', 'neptune', 'nmap', 'normal', 'perl', 'phf', 'pod', 'portsweep',

                  'rootkit', 'satan', 'smurf', 'spy', 'teardrop', 'warezclient', 'warezmaster']



train_label = pd.DataFrame()

test_label = pd.DataFrame()



for attack_type in attack_classes:

    train_label[attack_type] = train['attack_class'].apply(lambda x : int(x == attack_type))

    test_label[attack_type] = test['attack_class'].apply(lambda x : int(x == attack_type))
# extracting numerical labels from categorical data



encoder = LabelEncoder()



train['protocol_type_label'] = encoder.fit_transform(train['protocol_type'])

test['protocol_type_label'] = encoder.fit_transform(test['protocol_type'])



train['service_label'] = encoder.fit_transform(train['service'])

test['service_label'] = encoder.fit_transform(test['service'])



train['flag_label'] = encoder.fit_transform(train['flag'])

test['flag_label'] = encoder.fit_transform(test['flag'])
# removing useless columns



train.drop(['attack_class', 'num_learners'], axis = 1, inplace = True)

test.drop(['attack_class', 'num_learners'], axis = 1, inplace = True)
print("Training data information...")

train.info()
# creating dataframes for storing training data for stacked model



stacked_train_df = {}

stacked_test_df = {}



for attack_type in attack_classes:

    stacked_train_df[attack_type] = pd.DataFrame()

    stacked_test_df[attack_type] = pd.DataFrame()
# preparing data for training on models



x_train = train.copy(deep = True)

x_train.drop(['protocol_type', 'service', 'flag'], axis = 1, inplace = True)



x_test = test.copy(deep = True)

x_test.drop(['protocol_type', 'service', 'flag'], axis = 1, inplace = True)
# logistic regression classifier



def getLRClf():

    clf = LogisticRegression(C = 0.2, solver = 'liblinear')

    return clf
# training on logistic regression classifier



lr_accuracy = []



for attack_type in attack_classes:

    clf = getLRClf()

    clf.fit(x_train, train_label[attack_type])

    y_pred = clf.predict(x_test)

    stacked_train_df[attack_type]['logistic_regression'] = clf.predict(x_train)

    stacked_test_df[attack_type]['logistic_regression'] = y_pred

    lr_accuracy += [accuracy_score(test_label[attack_type], y_pred)]

    

mean_lr_accuracy = np.mean(lr_accuracy)

    

print("Logistic Regression Classifier...")

print("Mean Accuracy score : " + str(mean_lr_accuracy))
# SGD classifier



def getSGDClf():

    clf = SGDClassifier(max_iter = 1000, tol = 1e-3, learning_rate = 'optimal')

    return clf
# training on SGD classifier



sgd_accuracy = []



for attack_type in attack_classes:

    clf = getSGDClf()

    clf.fit(x_train, train_label[attack_type])

    y_pred = clf.predict(x_test)

    stacked_train_df[attack_type]['sgd'] = clf.predict(x_train)

    stacked_test_df[attack_type]['sgd'] = y_pred

    sgd_accuracy += [accuracy_score(test_label[attack_type], y_pred)]

    

mean_sgd_accuracy = np.mean(sgd_accuracy)

    

print("SGD Classifier...")

print("Mean Accuracy score : " + str(mean_sgd_accuracy))
# lgbm classifier



import lightgbm as lgb



def getlgbclf(d_train, valid_sets):

    params = {'learning_rate': 0.2, 'application': 'binary', 'num_leaves': 31, 'verbosity': -1,

          'bagging_fraction': 0.8, 'feature_fraction': 0.6, 'nthread': 4, 'lambda_l1': 1, 'lambda_l2': 1}

    

    clf = lgb.train(params, train_set = d_train, num_boost_round = 300, early_stopping_rounds = 100,

                    valid_sets = valid_sets, verbose_eval = False)   

    

    return clf
# training on lgbm classifier



lgb_accuracy = []



for attack_type in attack_classes:

    d_train = lgb.Dataset(x_train, label = train_label[attack_type])

    d_test = lgb.Dataset(x_test, label = test_label[attack_type])

    valid_sets = [d_train, d_test]

    clf = getlgbclf(d_train, valid_sets)

    y_pred = (clf.predict(x_test) >= 0.5).astype(int)

    stacked_train_df[attack_type]['lgbm'] = (clf.predict(x_train) >= 0.5).astype(int)

    stacked_test_df[attack_type]['lgbm'] = y_pred

    lgb_accuracy += [accuracy_score(test_label[attack_type], y_pred)]

    

mean_lgb_accuracy = np.mean(lgb_accuracy)

    

print("LGBM Classifier...")

print("Mean Accuracy score : " + str(mean_lgb_accuracy))
# XGBoost classifier



import xgboost as xgb



def getxgbclf(d_train, eval_list):

    params = {'booster' : 'gbtree', 'nthread' : 4, 'eta' : 0.2, 'max_depth' : 6, 'min_child_weight' : 4,

          'subsample' : 0.7, 'colsample_bytree' : 0.7, 'objective' : 'binary:logistic'}



    clf = xgb.train(params, d_train, num_boost_round = 300, early_stopping_rounds = 100, 

                    evals = evallist, verbose_eval = False)

    return clf
# training on XGBoost classifier



xgb_accuracy = []



for attack_type in attack_classes:

    d_train = xgb.DMatrix(x_train, label = train_label[attack_type])

    d_test = xgb.DMatrix(x_test, label = test_label[attack_type])

    evallist = [(d_train, 'train'), (d_test, 'valid')]

    clf = getxgbclf(d_train, evallist)

    y_pred = (clf.predict(d_test) >= 0.5).astype(int)

    stacked_train_df[attack_type]['xgb'] = (clf.predict(d_train) >= 0.5).astype(int)

    stacked_test_df[attack_type]['xgb'] = y_pred

    xgb_accuracy += [accuracy_score(test_label[attack_type], y_pred)]

    

mean_xgb_accuracy = np.mean(xgb_accuracy)

    

print("XGBoost Classifier...")

print("Mean Accuracy score : " + str(mean_xgb_accuracy))
# Deep Neural Network classifier



def getdnnclf():

    clf = Sequential()

    clf.add(Dense(1024, input_dim = 41, activation = 'relu'))

    clf.add(BatchNormalization())

    clf.add(Dense(1024, activation = 'relu'))

    clf.add(BatchNormalization())

    clf.add(Dense(512, activation = 'relu'))

    clf.add(Dense(64, activation = 'relu'))

    clf.add(Dense(1, activation = 'sigmoid'))

    clf.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

    return clf
# training on DNN classifier



dnn_accuracy = []



for attack_type in attack_classes:

    clf = getdnnclf()

    clf.fit(x_train, train_label[attack_type], batch_size = 1024, epochs = 5, 

            validation_data = (x_test, test_label[attack_type]), verbose = 0)

    y_pred = (clf.predict(x_test) >= 0.5).astype(int)

    stacked_train_df[attack_type]['dnn'] = (clf.predict(x_train) >= 0.5).astype(int)

    stacked_test_df[attack_type]['dnn'] = y_pred

    dnn_accuracy += [accuracy_score(test_label[attack_type], y_pred)]

    

mean_dnn_accuracy = np.mean(dnn_accuracy)

    

print("Deep Neural Network Classifier...")

print("Mean Accuracy score : " + str(mean_dnn_accuracy))
# training on stacked classifier



stacked_accuracy = []



for attack_type in attack_classes:

    clf = getLRClf()

    clf.fit(stacked_train_df[attack_type], train_label[attack_type])

    y_pred = clf.predict(stacked_test_df[attack_type])

    stacked_accuracy += [accuracy_score(test_label[attack_type], y_pred)]

    

mean_stacked_accuracy = np.mean(stacked_accuracy)

    

print("Stacked Classifier...")

print("Mean Accuracy score : " + str(mean_stacked_accuracy))
# models comparison



columns = ['Average Accuracy Score']

rows = ['Log Reg', 'SGD', 'LGBM', 'XGBoost', 'DNN', 'Stacked Model']

scores = [[mean_lr_accuracy], [mean_sgd_accuracy], [mean_lgb_accuracy], [mean_xgb_accuracy], 

          [mean_dnn_accuracy], [mean_stacked_accuracy]]



table = pd.DataFrame(data = scores, columns = columns, index = rows)

print(table)
# graphical comparison



n_groups = 6

acc = [item[0]-0.95 for item in scores]



fig, ax = plt.subplots()

index = np.arange(n_groups)

bar_width = 0.30

opacity = 0.8

 

rects = plt.bar(index, acc, bar_width, alpha = opacity, align = 'center', label = 'Average Accuracy')



plt.xlabel('Model')

plt.ylabel('Average Accuracy Score')

plt.title('Graphical Comparison of performances of all models')

plt.xticks(index + bar_width, rows)

plt.yticks([0.00, 0.01, 0.02, 0.03, 0.04, 0.05], [0.95, 0.96, 0.97, 0.98, 0.99, 1.00])

plt.legend()



fig = plt.tight_layout(rect = (0, 0, 1.4, 1.4))

plt.show()