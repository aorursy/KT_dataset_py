import numpy as np

import pandas as pd

import os

from sklearn.metrics import mean_squared_error, classification_report,confusion_matrix

import ast, math

import seaborn as sns

from collections import Counter

from shapely.geometry import Point

import geopandas as gpd

from geopandas import GeoDataFrame

import plotly.graph_objs as go

from sklearn.cluster import KMeans

import plotly.offline as py

from matplotlib.pyplot import figure

from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin,clone

import catboost as cat

from sklearn.neighbors import KNeighborsRegressor

from sklearn.mixture import GaussianMixture

from sklearn.impute import SimpleImputer

from sklearn.feature_selection import RFECV, RFE

import xgboost as xgb

import time

import datetime

from sklearn.decomposition import PCA

import eli5

from sklearn.feature_extraction.text import TfidfVectorizer

from scipy.special import boxcox1p,inv_boxcox1p,boxcox

from sklearn.preprocessing import RobustScaler,MinMaxScaler,StandardScaler,Normalizer, OneHotEncoder, LabelEncoder, PolynomialFeatures, Binarizer

from sklearn.pipeline import make_pipeline

from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier,RandomForestRegressor

from sklearn.model_selection import KFold, cross_val_score, train_test_split, StratifiedKFold

from statistics import median, stdev

from sklearn.linear_model import Lasso, ElasticNet, LinearRegression, Ridge, LogisticRegression, LassoLars

from sklearn.tree import DecisionTreeClassifier

from sklearn.kernel_ridge import KernelRidge

import lightgbm as lgb

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



import matplotlib.pyplot as plt

pd.options.mode.chained_assignment = None
train = pd.read_csv("../input/genpact/train.csv")

test = pd.read_csv("../input/genpact/test.csv")

meal_info = pd.read_csv("../input/genpact/meal_info.csv")

fulfilment_info = pd.read_csv("../input/genpact/fulfilment_center_info.csv")
train['city_code']=0

train['region_code']=0

train['center_type']='a'

train['op_area(km*km)']=0

test['city_code']=0

test['region_code']=0

test['center_type']='a'

test['op_area(km*km)']=0



index = fulfilment_info.index.values

center_id = fulfilment_info['center_id']

fulfilment_map = dict(zip(center_id, index))





for i in range(0,len(train)):

    train['city_code'][i]=fulfilment_info['city_code'][fulfilment_map[train['center_id'][i]]]

    train['region_code'][i]=fulfilment_info['region_code'][fulfilment_map[train['center_id'][i]]]

    train['center_type'][i]=fulfilment_info['center_type'][fulfilment_map[train['center_id'][i]]]

    train['op_area(km*km)'][i]=fulfilment_info['op_area'][fulfilment_map[train['center_id'][i]]]



    

for i in range(0,len(test)):

    test['city_code'][i]=fulfilment_info['city_code'][fulfilment_map[test['center_id'][i]]]

    test['region_code'][i]=fulfilment_info['region_code'][fulfilment_map[test['center_id'][i]]]

    test['center_type'][i]=fulfilment_info['center_type'][fulfilment_map[test['center_id'][i]]]

    test['op_area(km*km)'][i]=fulfilment_info['op_area'][fulfilment_map[test['center_id'][i]]]

train['category']='a'

train['cuisine']='a'

test['category']='a'

test['cuisine']='a'



index = meal_info.index.values

meal_id = meal_info['meal_id']

meal_map = dict(zip(meal_id, index))



for i in range(0,len(train)):

        train['category'][i] = meal_info['category'][meal_map[train['meal_id'][i]]]

        train['cuisine'][i] = meal_info['cuisine'][meal_map[train['meal_id'][i]]]



for i in range(0,len(test)):

        test['category'][i] = meal_info['category'][meal_map[test['meal_id'][i]]]

        test['cuisine'][i] = meal_info['cuisine'][meal_map[test['meal_id'][i]]]
test.to_csv('test.csv', index=False)

train.to_csv('train.csv', index=False)