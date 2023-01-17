import numpy as np

import pandas as pd

pd.set_option('max_columns', None)

import matplotlib.pyplot as plt

plt.style.use('ggplot')

import datetime

from scipy import stats

from scipy.sparse import hstack, csr_matrix

from sklearn.model_selection import train_test_split, KFold

from wordcloud import WordCloud

from collections import Counter

from nltk.corpus import stopwords

from nltk.util import ngrams

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.preprocessing import StandardScaler,MinMaxScaler, Imputer,LabelEncoder

from sklearn.metrics import roc_auc_score



import gc

import re

import seaborn as sns

import os

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls

import xgboost as xgb

import lightgbm as lgb

from sklearn.linear_model import LogisticRegression

from sklearn import model_selection

from sklearn.metrics import accuracy_score

import json

import ast

# import shap

from urllib.request import urlopen

from PIL import Image

import time

from sklearn.metrics import mean_squared_error
test = pd.read_csv('../input/digix-test/test_20190518.csv',

                   names=['num','uId','adId','operTime','siteId','slotId','contentId','netType'])

for col in test:

    if test[col].dtype == 'int64':

        test[col] = test[col].astype(np.int32)

train = pd.read_csv('../input/digix-train/train_20190518.csv',

                    names=['label','uId','adId','operTime','siteId','slotId','contentId','netType'],nrows=15000000)        

for col in train:

    if train[col].dtype == 'int64':

        train[col] = train[col].astype(np.int32)
train_label_0 = train[ train['label'] == 0 ]

train_label_1 = train[ train['label'] == 1 ]

print(train_label_0.shape,train_label_1.shape)



train_label_0 = train_label_0.sample(n = 800000 , random_state = 42).reset_index(drop=True)

train_label_1 = train_label_1.sample(n = 800000 , random_state = 42).reset_index(drop=True)

print(train_label_0.shape,train_label_1.shape)



train=pd.concat([train_label_0,train_label_1]).reset_index(drop=True)

train = train.sample(frac = 1, random_state = 32).reset_index(drop=True)

print(train.shape)

print(train['label'].value_counts())
user_info = pd.read_csv('../input/digix-data/user_info.csv',

                        names=['uId','age','gender','city','province','phoneType','carrier'])

for col in user_info:

    if user_info[col].dtype == 'int64':

        user_info[col] = user_info[col].astype(np.int32)

ad_info = pd.read_csv('../input/digix-data/ad_info.csv',

                      names=['adId','billId','primId','creativeType','intertype','spreadAppId'])

       

content_info = pd.read_csv('../input/digix-data/content_info.csv',

                           names=['contentId','firstClass','secondClass'])



train = pd.merge(train, ad_info, how='left', on=['adId'])

train = pd.merge(train, content_info, how='left', on=['contentId'])

train = pd.merge(train, user_info, how='left', on=['uId'])

test = pd.merge(test, ad_info, how='left', on=['adId'])

test = pd.merge(test, content_info, how='left', on=['contentId'])

test = pd.merge(test, user_info, how='left', on=['uId'])



train.to_csv('train.csv', index = False)

test.to_csv('test.csv', index = False)