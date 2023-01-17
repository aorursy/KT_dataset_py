import numpy as np

import pandas as pd

import csv

from datetime import datetime

import re

import math

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px



%matplotlib inline
data = pd.read_csv("../input/nba2k20-player-dataset/nba2k20-full.csv")
def clean_data(data):

    data['salary'] = data['salary'].apply(lambda x: int(x[1:]))

    data['jersey'] = data['jersey'].apply(lambda x: int(x[1:]))

    data['b_day'] = data['b_day'].apply(lambda x: datetime.strptime(x, '%m/%d/%y').date())

    data['height'] = data['height'].apply(lambda x: float(x[2+x.find('/'):]))

    data['weight'] = data['weight'].apply(lambda x: float(x[2+x.find('/'):-4]))

    data['draft_round'] = data['draft_round'].apply(lambda x: int(x) if len(x) == 1 else 0)

    data['draft_peak'] = data['draft_peak'].apply(lambda x: int(x) if 1<=len(x)<=2 else 0)

    data['college'] = data['college'].fillna('no education')

    data['team'] = data['team'].fillna('no team')



clean_data(data)
#find age of each player



def age_(birthday):

    today = datetime.strptime(datetime.today().strftime('%Y-%m-%d'), '%Y-%m-%d').date()

    age = today.year - birthday.year

    return int(age)



data['age'] = data['b_day'].apply(lambda x: age_(x))

data.loc[data['country'] != 'USA', 'country'] = 'not USA'

data.loc[data['position'] == 'F-G', 'position'] = 'F'

data.loc[data['position'] == 'G-F', 'position'] = 'F'

data.loc[data['position'] == 'F-C', 'position'] = 'C'

data.loc[data['position'] == 'C-F', 'position'] = 'C'

data
from sklearn import preprocessing

from scipy import stats



data_dummy = pd.get_dummies(data, columns=['team', 'position','draft_round', 'country'], drop_first= True)

data_dummy = data_dummy.drop(['full_name', 'draft_peak', 'b_day', 'jersey', 'college'], axis = 1)

X, y = data_dummy.drop(['salary'], axis = 1), data_dummy['salary']

data_dummy
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error



# import tensorflow as tf

# from tensorflow import keras

from sklearn import preprocessing

from xgboost import XGBRegressor

# from sklearn.neighbors import KNeighborsRegressor
normalizer = preprocessing.Normalizer().fit(X)

X = normalizer.transform(X)

X = np.array(X)

y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
model = model = XGBRegressor( 

    learning_rate=0.04,

    colsample_bytree=0.9, 

    min_child_weight=3.5,

    objective='reg:squarederror',

    max_depth = 2,

    subsample = 0.63,

    eta = 0.1,

    seed=0)



model = model.fit(

    X_train, 

    y_train, 

    eval_metric="rmse", 

    verbose=True)
predictions = model.predict(X_test)

predictions
np.sqrt(mean_squared_error(y_test, predictions))