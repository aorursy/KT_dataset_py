# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.impute import SimpleImputer

import os

import pickle

from sklearn.preprocessing import StandardScaler,MinMaxScaler,Imputer

from sklearn.metrics import confusion_matrix, roc_curve,roc_auc_score,classification_report

from sklearn.metrics import accuracy_score

from sklearn.model_selection import *

import keras

from keras.preprocessing import sequence

from keras.utils import to_categorical

from sklearn.preprocessing import LabelEncoder

from keras.layers import LSTM, Dense, Input, TimeDistributed, Masking

from keras.models import Model, Sequential

from keras.optimizers import RMSprop

from keras import layers

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import itertools

import os

from keras.models import load_model

from sklearn.externals import joblib

from keras.callbacks import EarlyStopping

import tensorflow as tf

from keras import backend as K

from scipy import stats

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import *

from sklearn.linear_model import LogisticRegression

from sklearn.feature_selection import RFECV

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import *

import xgboost as xgb;



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/equipfails/equip_failures_training_set.csv')

test = pd.read_csv('../input/equipfails/equip_failures_test_set.csv')

sample_submission = pd.read_csv('../input/equipfails/sample_submission.csv')
train = train.replace('na', np.NaN)

test = test.replace('na', np.NaN)
test.columns
column_names = list(train.columns)

train_selected_columns = column_names[2:]

train_predictors = train[train_selected_columns]

train_target = train['target']

test_predictors = test[train_selected_columns]
test_predictors
train_surface = train.loc[train['target'] == 1]

train_downhole = train.loc[train['target'] == 0]

train_downhole_undersample = train_downhole.sample(n=2000)

df_frames = [train_surface, train_downhole_undersample]

train_concact = pd.concat(df_frames)
import plotly.express as px



fig = px.parallel_categories(train_concact, color="target")

df_Y = train_concact['target']

col_names = train_concact.columns

predictor_names = col_names[2:]

df_x = train_concact[predictor_names]
y = df_Y.apply(pd.to_numeric)

X = df_x.apply(pd.to_numeric)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
# Impute our data, then train

X_train_imp = imp.transform(X_train)

clf = RandomForestClassifier(n_estimators=10)

clf = clf.fit(X_train_imp, y_train)



X_test_imp = imp.transform(X_test)
print(X_test, '->', clf.predict(X_test_imp))