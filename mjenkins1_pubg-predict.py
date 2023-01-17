import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns 

import os

import warnings

import pickle

import gc



from sklearn.metrics import mean_absolute_error as mae

from sklearn.model_selection import train_test_split

import lightgbm as lgb



%matplotlib inline

warnings.filterwarnings("ignore")
# Set the size of the plots 

plt.rcParams["figure.figsize"] = (18,8)

sns.set(rc={'figure.figsize':(18,8)})
model = pickle.load(open('../input/pubg-presentation-model/model_lgbm.pkl', 'rb'))

print("Model loaded")
train = pd.read_csv("../input/pubg-presentation-features-engineering/train.csv")

test = pd.read_csv("../input/pubg-presentation-features-engineering/test.csv")

print("Finished loading the data")
test.head()
test.columns
y_test = test['winPlacePerc']
X_test = test

X_test.drop(['Unnamed: 0', 'winPlacePerc'], inplace=True, axis=1)
pred = model.predict(X_test)

mae_scr = mae(y_test, pred)

print("SCORE:", mae_scr)