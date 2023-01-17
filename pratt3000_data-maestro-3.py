from bayes_opt import BayesianOptimization

from xgboost import XGBRegressor

from sklearn.model_selection import StratifiedKFold, KFold

from sklearn.preprocessing import LabelEncoder

from sklearn import metrics



###############################################

# Import Miscellaneous Assets

###############################################

# Data manipulation

import pandas as pd

import numpy as np



# Modeling

import lightgbm as lgb



# Evaluation of the model

from sklearn.model_selection import KFold, train_test_split

from sklearn.metrics import roc_auc_score



# Visualization

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

from pandas.plotting import scatter_matrix

%matplotlib inline





from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2



from dateutil.parser import parse

from datetime import datetime

from scipy.stats import norm



# import all what you need for machine learning

import sklearn

from sklearn.model_selection import cross_val_predict

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import cross_validate

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import RandomizedSearchCV

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import RobustScaler

import warnings

from datetime import datetime

from functools import partial

from pprint import pprint as pp

from tqdm import tqdm, tqdm_notebook



import tensorflow as tf

import math

from hyperopt import hp



from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix

from sklearn.externals import joblib



import xgboost as xgb

from sklearn.metrics import accuracy_score



from numpy import loadtxt

from xgboost import XGBClassifier



import lightgbm as lgb

from hyperopt import STATUS_OK



from keras.models import Sequential

from keras.layers import Dense, Activation, Dropout

from keras import optimizers



from sklearn.metrics import accuracy_score

import os

from sklearn import preprocessing



import h2o

from h2o.estimators import H2ORandomForestEstimator

from h2o.estimators import H2OXGBoostEstimator

###### ESSIENTIAL CODE ###########

train = pd.read_csv("/kaggle/input/datamaestro2020/astro_train.csv")

test = pd.read_csv("/kaggle/input/datamaestro2020/astro_test.csv")

sample = pd.read_csv("/kaggle/input/datamaestro2020/sample_submission.csv")



### useless thing ifigured all out in different notebook, data_maestro_1 dekho ab

del train["id"]

del test["id"]



del train["rerun"]

del test["rerun"]

del train["skyVersion"]

del test["skyVersion"]

del train["run"]

del test["run"]

del train["camCol"]

del test["camCol"]



### minmax scaling karna hai but it gets converted to a different data type 

### bhavika tera manually kiya hua function bhej lol



train.to_csv('train_new.csv', header=True, index=False)

test.to_csv('test_new.csv', header=True, index=False)
import h2o

from h2o.estimators.gbm import H2OGradientBoostingEstimator

from bayes_opt import BayesianOptimization

h2o.init()

h2o.remove_all()



data = h2o.upload_file("/kaggle/working/train_new.csv")

train_cols = [x for x in data.col_names if x not in ['class']]

target = "class"

train, test = data.split_frame(ratios=[0.7])
def train_model(learn_rate, 

                max_depth,

                min_child_weight, 

                gamma,

                colsample_bytree,

                eta,

                ntrees, 

                ):

    params = {

        'learn_rate': learn_rate ,

        'max_depth' :int(max_depth) ,

        'min_child_weight' :int(min_child_weight) ,

        'gamma' : gamma,

        'colsample_bytree' : colsample_bytree,

        'eta':eta,

        'ntrees': int(ntrees),

        'subsample': 0.8,

        'booster' : ("dart"),

        'dmatrix_type' : ("sparse"),

        'tree_method' : ("hist"),

        'sample_type' : ("weighted"),

        'normalize_type' : ("forest")

    }

    model = H2OXGBoostEstimator(nfolds=5,**params)

    model.train(x=train_cols, y=target, training_frame=train)

    return -model.rmse()
bounds = {

        'learn_rate': (0.01,0.3) ,

        'max_depth' :(12,20) ,

        'min_child_weight' :(1,4) ,

        'gamma' : (0.01,0.5),

        'colsample_bytree' : (0.3,1.0),

        'eta':(0.01,0.2),

        'ntrees': (160,200),

    }
optimizer = BayesianOptimization(

    f=train_model,

    pbounds=bounds,

    random_state=1,

)

optimizer.maximize(init_points=10, n_iter=50)
optimizer.max
{'colsample_bytree': 1.0,

  'eta': 0.2,

  'gamma': 0.0,

  'learn_rate': 0.3,

  'max_depth': 15.0,

  'min_child_weight': 1.0,

  'ntrees': 178.7243273681921}
#set manually to optimiser.max idk how to do directly cheap trick lol

#maxdepth and ntrees int karaychi approx



#defining

model = H2OXGBoostEstimator(colsample_bytree=1, 

                                     eta = 0.2,

                                     gamma  = 0.01,

                                     learn_rate = 0.01,

                                     max_depth = 15,

                                     min_child_weight =1,

                                     ntrees = 163

                                    )



#traneeeee

model.train(x=train_cols, y=target, training_frame=train)
#a lotta random convesions here which i tuk too long to figar out

test = h2o.upload_file("/kaggle/working/test_new.csv")



predictions = model.predict(test)

pdr = predictions.as_data_frame().to_numpy()

p = np.round(pdr).astype(int)



#allah ho gaya finally



submission_df = pd.DataFrame(columns=['id', 'class'])

submission_df['id'] = sample['id']

submission_df['class'] = p

submission_df.to_csv('bayesian_final.csv', header=True, index=False)



#dekhlo ab kyahi, test noi banaya banana hai toh use own brain

submission_df.head(10)