# Pandas / Numpy for data loading and preparation

import pandas as pd

import numpy as np

import scipy as sp

import sklearn as sk

from scipy.sparse import csr_matrix, hstack



# Charts / graphs

import matplotlib.pyplot as plt

from __future__ import division

from mpl_toolkits.basemap import Basemap

from matplotlib.colors import LogNorm

import matplotlib.pylab as pylab

import seaborn as sns



# General purpose classes / utilities

from copy import deepcopy

import json

import timeit

import datetime

import time

import os

from ast import literal_eval

import csv

import ast

import itertools



# NLTK

import nltk.corpus

from nltk import SnowballStemmer



# Scipy

from scipy.stats import skew, boxcox



# Scikit-learn

from sklearn import preprocessing

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import LabelEncoder



from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import KFold

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_selection import SelectPercentile, f_classif, chi2



from sklearn.svm import SVC

from sklearn.cluster import DBSCAN

from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import Ridge

from sklearn.ensemble import RandomForestClassifier



from sklearn.metrics import log_loss

from sklearn.metrics import make_scorer

from sklearn.metrics import roc_auc_score

from sklearn.metrics import mean_absolute_error



# Geopy for geographic clustering

from geopy.distance import great_circle



# Keras framework for neural network training

from keras.layers.advanced_activations import PReLU

from keras.layers.core import Dense, Dropout, Activation

from keras.constraints import maxnorm

from keras.layers.normalization import BatchNormalization

from keras.models import Sequential

from keras.utils import np_utils

from keras.models import model_from_json

from keras.utils.np_utils import to_categorical

from keras.wrappers.scikit_learn import KerasClassifier



# XGBoost framework

import xgboost as xgb



%matplotlib inline 
# Location of files and basic identifiers

ID = 'id'

TIMESTAMP = 'timestamp'

TARGET = 'y'

SEED = 0

DATA_DIR = "../input"

TRAIN = "{0}/train.h5".format(DATA_DIR)

TEST = "{0}/train.h5.csv".format(DATA_DIR)

SUBMISSION = "{0}/sample_submission.csv".format(DATA_DIR)



# read data

with pd.HDFStore(TRAIN, "r") as train:

    df = train.get("train")
# print all rows and columns

pd.set_option('display.max_columns', None)



# basic stats

print('The training set contains:')

print('{} records'.format(df.shape[0]))

print('{} features'.format(df.shape[1]))



# check colums types and values

features = list(df.columns)

derived_features = [x for x in features if x.find('derived') != -1]

fundamental_features = [x for x in features if x.find('fundamental') != -1]

technical_features = [x for x in features if x.find('technical') != -1]



print ('\nFeature FAMILIES:')

print ('- {} derived features'.format(len(derived_features)))

print ('- {} fundamental features'.format(len(fundamental_features)))

print ('- {} technical features'.format(len(technical_features)))

print('\nFeature TYPES:')

print('{}'.format(df.dtypes.value_counts()))



print('\n{} distinct time series each with an average of {} datapoints over a total time span of {} periods'.format(df.id.nunique(), 

                                                                             int(np.round(df.shape[0]/df.id.nunique(),0)),

                                                                             len(df.timestamp.unique())))
market = df[['timestamp', 'y']].groupby('timestamp').agg([np.mean, np.std, len]).reset_index()

t      = market['timestamp']

y_mean = np.array(market['y']['mean'])

y_std  = np.array(market['y']['std'])

n      = np.array(market['y']['len'])
plt.figure()

plt.plot(t, y_std, '-')

plt.xlabel('timestamp')

plt.ylabel('std of y')

plt.title('Portfolio VOLATILITY over time')
plt.figure()

plt.plot(t, y_mean, '-')

plt.xlabel('timestamp')

plt.ylabel('mean of y')

plt.title('Portfolio AVERAGE VALUE over time')
plt.figure()

plt.plot(t, y_std, '-')

plt.xlabel('timestamp')

plt.ylabel('std of y')

plt.title('Portfolio VOLATILITY over time')
plt.figure()

plt.plot(t, n, '-')

plt.xlabel('timestamp')

plt.ylabel('portfolio size')

plt.title('Portfolio ASSET COUNT over time')
import kagglegym

env = kagglegym.make()

observation = env.reset()
# Get the train dataframe

train = observation.train

mean_values = train.mean(axis=0)

# median_values = train.median(axis=0)

train.fillna(mean_values, inplace=True)





cols_to_use = ['technical_30', 'technical_20', 'fundamental_11']

# Observed with histograns:

low_y_cut = -0.086093

high_y_cut = 0.093497



y_is_above_cut = (train.y > high_y_cut)

y_is_below_cut = (train.y < low_y_cut)

y_is_within_cut = (~y_is_above_cut & ~y_is_below_cut)
def get_weighted_y(series):

    id, y = series["id"], series["y"]

    # return 0.95 * y + 0.05 * ymean_dict[id] if id in ymean_dict else y

    return 0.95 * y + 0.05 * ymedian_dict[id] if id in ymedian_dict else y
model = Ridge()

model.fit(np.array(train.loc[y_is_within_cut, cols_to_use].values), train.loc[y_is_within_cut, TARGET])



# ymean_dict = dict(train.groupby(["id"])["y"].mean())

ymedian_dict = dict(train.groupby(["id"])["y"].median())
while True:

    

    # make prediction

    observation.features.fillna(mean_values, inplace=True)

    test_x = np.array(observation.features[cols_to_use].values)

    observation.target.y = model.predict(test_x).clip(low_y_cut, high_y_cut)

    # weighted y using average value

    observation.target.y = observation.target.apply(get_weighted_y, axis = 1)

    

    # Execute step

    target = observation.target

    timestamp = observation.features["timestamp"][0]

    if timestamp % 100 == 0:

        print("Timestamp #{}".format(timestamp))



    observation, reward, done, info = env.step(target)

    if done:        

        break
print(info)