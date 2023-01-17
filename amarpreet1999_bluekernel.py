import numpy as np 

import pandas as pd



from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error as mse

from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LinearRegression

from sklearn.neural_network import MLPRegressor



import os
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv('/kaggle/input/bluebook-for-bulldozers/trainandvalid/TrainAndValid.csv')

test = pd.read_csv('/kaggle/input/bluebook-for-bulldozers/Test.csv')
train.info()
test.info()
train.head()
train.columns
train.describe(include='all')
train.isna().sum()
train.fiProductClassDesc
train['SalePrice'] = np.log(train.SalePrice)
features_to_consider = ['YearMade', 'datasource', 'state', 'fiBaseModel', 'fiProductClassDesc' , 'fiModelDesc']
def model_score(model, X_trn, y_trn, X_val, y_val):

    model.fit(X_trn, y_trn)

    pred = model.predict(X_val)

    return np.sqrt(mse(pred, y_val))
X = train[features_to_consider]

y = train.SalePrice
LabelEnc = LabelEncoder()

X['state']=LabelEnc.fit_transform(X.state)

X['fiBaseModel']= LabelEnc.fit_transform(X.fiBaseModel)

X['fiProductClassDesc']= LabelEnc.fit_transform(X.fiProductClassDesc)

X['fiModelDesc']= LabelEnc.fit_transform(X.fiModelDesc)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
model = LinearRegression()

model_score(model, X_train, y_train, X_test, y_test)
model = RandomForestRegressor(max_depth=30, min_samples_split=20, n_estimators=110, n_jobs= -1)

model_score(model, X_train, y_train, X_test, y_test)
model = MLPRegressor(hidden_layer_sizes=(100), activation="relu", solver="adam", alpha=0.0001, verbose=True)

model_score(model, X_train, y_train, X_test, y_test)
!pip install git+https://github.com/fastai/fastai@2e1ccb58121dc648751e2109fc0fbf6925aa8887
import os



import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

from pandas_summary import DataFrameSummary

from sklearn import metrics

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from time import time





## These are the fastai imports

from fastai.imports import *

from fastai.structured import *
data = pd.read_csv('/kaggle/input/bluebook-for-bulldozers/trainandvalid/TrainAndValid.csv', low_memory=False, parse_dates=["saledate"])

data.head()
train_cats(data)

data.head()
add_datepart(data, 'saledate')
data.head()
data.UsageBand = data.UsageBand.astype('category')

data.UsageBand = data.UsageBand.cat.codes
data.head()
data['SalePrice'] = np.log(data['SalePrice'])

data['SalePrice'].head()
finalData, Y, nas = proc_df(data, 'SalePrice')
finalData.head()
print(Y)

len(Y)
model = RandomForestRegressor(n_jobs=-1)
model.fit(finalData, Y)

model.score(finalData, Y)
X_train, X_test, y_train, y_test = train_test_split(finalData, Y, test_size=0.33, random_state=42)
model.fit(X_train, y_train)

model.score(X_test, y_test)
print(model.score(X_test, y_test) * 100)