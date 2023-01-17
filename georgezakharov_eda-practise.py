import pandas as pd

import numpy as np



import matplotlib.pyplot as plt



# Normalization tests

from scipy.stats import kurtosis

from scipy.stats import skew

from scipy.stats import shapiro

from scipy.stats import normaltest



# Data normalization / standartization

from sklearn.preprocessing import PowerTransformer, MinMaxScaler, StandardScaler



# libs for train tuning

from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split



# Boosting

from sklearn.ensemble import GradientBoostingClassifier

from xgboost.sklearn import XGBClassifier



# Classification

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

from sklearn.naive_bayes import BernoulliNB, ComplementNB, GaussianNB, MultinomialNB

from sklearn.svm import SVC



# Model parameters fitting

from sklearn.model_selection import GridSearchCV



# Metrics

from sklearn.metrics import accuracy_score,auc, f1_score, confusion_matrix,precision_score, recall_score, roc_auc_score, roc_curve



# Serialize data

import pickle



import random
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Get dataset

df = pd.read_csv('/kaggle/input/historical-data-on-the-trading-of-cryptocurrencies/crypto_tradinds.csv')
#Check the shape

df.shape
# Raw dataset

df.head()
# Dataset common info

df.info()
# Dataset common stats

df.describe(include='all')
df['crypto_type'].value_counts(normalize=True)
# Check nulls

df.isnull().sum()
# Functions for normalization test

def draw_hist_for_feature(data):

  plt.style.use('ggplot')

  data.hist(bins = 60)

  plt.show()





def chech_skew(feature):

  method_name = '\nSKEW TEST: '

  skew_ = np.abs(skew(feature))

  if (skew_ >= 0.75) and (skew_ < 1.0):

    print(method_name + 'Use logarithm method for data\n')

  elif skew_ >= 1:

    print(method_name + 'Use normalization method for data\n')

  else:

    print(method_name + 'Use standartization method for data\n')





def check_shapiro(feature):

  method_name = '\nSHAPIRO TEST: '

  shapiro_ = np.abs(shapiro(feature))

  if (shapiro_[1] < 0.50):

    print(method_name + 'Use normalization method for data\n')

  else:

    print(method_name + 'Use standartization method for data\n')





def print_stats(data, need_hist = True):

  if (need_hist == True):

    draw_hist_for_feature(data)



  print("mean : ", np.mean(data))

  print("var  : ", np.var(data))

  print("skew : ", skew(data))

  print("kurt : ", kurtosis(data))

  print("shapiro : ", shapiro(data))

  print("normaltest : ", normaltest(data))





def print_stats_all(df, need_hist = True):

  n = 1

  for feature_name in df.columns:

    print(f'\n\n{n}. {feature_name}')

    print_stats(df[feature_name], need_hist)

    chech_skew(df[feature_name])

    check_shapiro(df[feature_name])

    n += 1
# Divide numerical and categorical data

X_num = df.drop(['trade_date', 'crypto_name', 'crypto_type', 'ticker', 'site_url', 'github_url', 'minable', 'platform_name', 'industry_name'], axis = 1)

X_cat = df[['trade_date', 'crypto_name', 'crypto_type', 'ticker', 'site_url', 'github_url', 'minable', 'platform_name', 'industry_name']]
# Check normalization

print_stats_all(X_num)