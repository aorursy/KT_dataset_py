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
train = pd.read_csv("../input/pubg-finish-placement-prediction/train_V2.csv", low_memory=False)
len(train)
def display_all(df):

    with pd.option_context("display.max_rows", 1500):

        with pd.option_context("display.max_column", 10000):

            display(df)

    
display_all(train.head().transpose())
from sklearn.ensemble import RandomForestRegressor

from pandas_summary import DataFrameSummary

from IPython.display import display
from pandas.api.types import is_string_dtype, is_numeric_dtype, is_categorical_dtype
m = RandomForestRegressor(n_jobs = -1)
train.drop(['Id', 'matchId','groupId'], axis =1, inplace= True)
'''def convert_cats(df):

    for n, c in df.items():

        if is_string_dtype(c):df[n]=c.astype("category").cat.as_ordered()'''

    
train.dropna(inplace = True)
#train.Id.cat.codes
!pip install feather-format
os.makedirs('tmp', exist_ok=True)
train.reset_index()
display_all(train.head().transpose())
train.to_csv('tmp/train.csv')
tt=pd.read_csv('tmp/train.csv')
tt.to_feather('tmp/train-raw')
trainn=pd.read_feather('tmp/train-raw')
y = trainn['winPlacePerc']
df = trainn.drop(['winPlacePerc'], axis =1)
df = pd.get_dummies(df, columns= ['matchType'], drop_first = True )

display_all(df.head())
m = RandomForestRegressor(n_jobs=-1)
m.fit(trainn, y)

m.score(trainn,y)
#convert_cats(df)
#df.Id.cat.codes
#df = df.drop(['Id'], axis =1)
#m.fit(df, y)
#y.dropna(inplace = True)
#df = train.drop(['winPlacePerc'], axis =1)
df.dtypes
#df.dropna(inplace=True)
def split_vals(a,n): return a[:n].copy(), a[n:].copy()

n_valid = 934174  # same as Kaggle's test set size

n_trn = len(df)-n_valid

raw_train, raw_valid = split_vals(train, n_trn)

X_train, X_valid = split_vals(df, n_trn)

y_train, y_valid = split_vals(y, n_trn)

X_train.shape, y_train.shape, X_valid.shape
def print_score(m):

    res = [mean_absolute_error(m.predict(X_train), y_train),

           mean_absolute_error(m.predict(X_valid), y_valid),

           m.score(X_train, y_train), m.score(X_valid, y_valid)]

    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)

    print(res)
from sklearn.metrics import mean_absolute_error
m = RandomForestRegressor(n_jobs=-1, n_estimators=20)

%time m.fit(X_train, y_train)

print_score(m)