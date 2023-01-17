import numpy as np
import pandas as pd
import gc

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

from IPython.display import display # Allows the use of display() for DataFrames

import warnings
warnings.filterwarnings('ignore')
# Read train and test files
train_df = pd.read_csv('../input/santander-value-prediction-challenge/train.csv')
test_df = pd.read_csv('../input/santander-value-prediction-challenge/test.csv')
train_df.head()
