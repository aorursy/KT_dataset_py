# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import matplotlib.pyplot as plt

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

sns.set()



from sklearn.linear_model import LinearRegression

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

import xgboost
data = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

print("Data shape: ", data.shape)

print("Duplicates: ", data.duplicated().sum())

print()

print(data.isnull().sum())

print()

data.head()
df = data.copy()
numerics = ['int64', 'float64']

numerical = df[[c for c,v in df.dtypes.items() if v in numerics]]

categorical = df[[c for c,v in df.dtypes.items() if v not in numerics]]

for col in numerical.columns:

    numerical.fillna(numerical[col].mean(), inplace=True, axis=1)

X = numerical.drop(['Id', 'YearBuilt', 'YearRemodAdd', 'MoSold', 'YrSold', 'SalePrice'], axis=1)

Y = numerical[['SalePrice']]

print(X.shape, Y.shape)
scaler = StandardScaler()

scaler.fit(X)

X_ = scaler.transform(X)

X = pd.DataFrame(data=X_, columns = X.columns)

X.head()
xtrain ,xtest, ytrain, ytest = train_test_split(X.values, Y.values, test_size=0.25, random_state=12)

print(xtrain.shape, ytrain.shape)

print(xtest.shape, ytest.shape)
model = LinearRegression(fit_intercept=True, n_jobs =1)

model.fit(xtrain, ytrain)

pred_LinearRegression = model.predict(xtest)
def make_dataframe(pred, ytest, columns=5):

    pp = pred.tolist()

    ap = ytest.tolist()

    #pp = np.squeeze(pp, axis=1)

    #ap = np.squeeze(ap, axis=1)

    data = pd.DataFrame({

        'Predicted Price' : pp,

        'Actual Price' : ap

    })

    return data.head(columns)
make_dataframe(pred_LinearRegression, ytest)
def plot_predictions(pred, ytest):

    sns.set_context('poster')

    plt.figure(figsize=(12,8))

    plt.plot(ytest[:100], label='original');

    plt.plot(pred[:100], label='prediciton');

    plt.legend();

plot_predictions(pred_LinearRegression, ytest)
r2_LinearRegression = r2_score(pred_LinearRegression, ytest)

r2_LinearRegression
mse_LinearRegression = mean_squared_error(ytest, pred_LinearRegression)

mse_LinearRegression

#XGBRegressor

model = xgboost.XGBRegressor(colsample_bytree=0.4,

                 gamma=0,                 

                 learning_rate=0.001,

                 max_depth=3,

                 min_child_weight=1.5,

                 n_estimators=10000,                                                                    

                 reg_alpha=0.75,

                 reg_lambda=0.45,

                 subsample=0.6,

                 seed=42)

model.fit(xtrain, ytrain)

pred_XGBRegressor = model.predict(xtest)
make_dataframe(pred_XGBRegressor, ytest)
plot_predictions(pred_XGBRegressor, ytest)
r2_XGBRegressor = r2_score(pred_XGBRegressor, ytest)

r2_XGBRegressor

mse_XGBRegressor = mean_squared_error(ytest, pred_XGBRegressor)

mse_XGBRegressor
