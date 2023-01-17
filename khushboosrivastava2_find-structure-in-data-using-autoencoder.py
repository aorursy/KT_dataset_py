import numpy as np

import pandas as pd

import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from pylab import rcParams

rcParams['figure.figsize']=15,10
# Read data

train=pd.read_csv('../input/Kaggle_Training_Dataset_v2.csv', sep=',')

test=pd.read_csv('../input/Kaggle_Test_Dataset_v2.csv', sep=',')

train.head()

test.head()
train.dtypes
train.info(verbose=False)
#Categorical Analysis

train[train.select_dtypes(include = ['object']).columns].head()
train['went_on_backorder'].value_counts() #target variable
train.describe(include = 'all')
missing_data_train = train.isnull().sum().sort_values(ascending=False)

missing_data_train
missing_data_test = test.isnull().sum().sort_values(ascending=False)

missing_data_test
from sklearn.preprocessing import Imputer

train['lead_time'] = Imputer(strategy='median').fit_transform(

                               train['lead_time'].values.reshape(-1, 1))

train = train.dropna()

train.isnull().sum()
from sklearn.preprocessing import Imputer

train['lead_time'] = Imputer(strategy='median').fit_transform(

                              train['lead_time'].values.reshape(-1, 1))

train = train.dropna()

train.isnull().sum()
train.head()
from sklearn.preprocessing import Imputer

test['lead_time'] = Imputer(strategy='median').fit_transform(

                              test['lead_time'].values.reshape(-1, 1))

rest = test.dropna()

test.isnull().sum()
binaries = ['potential_issue', 'deck_risk', 'oe_constraint', 'ppap_risk',

               'stop_auto_buy', 'rev_stop', 'went_on_backorder']

for col in binaries:

      train[col] = (train[col] == 'Yes').astype(int)

train[binaries].head()
#H2o

import h2o

from h2o.estimators.gbm import H2OGradientBoostingEstimator

from h2o.estimators.deeplearning import H2OAutoEncoderEstimator

h2o.init(nthreads = -1)
train_h2o = h2o.H2OFrame(train)

test_h2o = h2o.H2OFrame(test)



#  Identify predictors and response

X = train_h2o.columns          

y = "went_on_backorder"    # Target column name

X.remove(y)
# For binary classification, response should be a factor

train_h2o['went_on_backorder'] = train_h2o['went_on_backorder'].asfactor()

test_h2o['went_on_backorder'] = test_h2o['went_on_backorder'].asfactor()
anomaly_model = H2OAutoEncoderEstimator(   activation="Tanh",

                                           hidden=[25,  2,  25],

                                           ignore_const_cols = False,

                                           epochs=500)



anomaly_model.train(x = X,  training_frame = train_h2o)

recon_error = anomaly_model.anomaly(train_h2o)



# Get MSE only

print("MSE := ",anomaly_model.mse())
layerLevel= 1

bidimensional_data = anomaly_model.deepfeatures(train_h2o,layerLevel)

bidimensional_data = bidimensional_data.cbind(train_h2o['went_on_backorder'])

bidimensional_data = bidimensional_data.as_data_frame()



sns.FacetGrid(bidimensional_data, hue="went_on_backorder", size=10).map(plt.scatter, "DF.L2.C1", "DF.L2.C2").add_legend()

plt.show()