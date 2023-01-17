import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib

from sklearn import preprocessing 

import matplotlib.pyplot as plt

from scipy.stats import skew

from scipy.stats.stats import pearsonr





%config InlineBackend.figure_format = 'retina' #set 'png' here when working on notebook

%matplotlib inline
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.head()
#merge all data

all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],

                      test.loc[:,'MSSubClass':'SaleCondition']))

all_data.head()
matplotlib.rcParams['figure.figsize'] = (12.0, 6.0)

prices = pd.DataFrame({"price":train["SalePrice"], "log(price + 1)":np.log1p(train["SalePrice"])})

prices.hist()
#log transform the target:

train["SalePrice"] = np.log1p(train["SalePrice"])



#log transform skewed numeric features:

numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index



skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness

skewed_feats = skewed_feats[skewed_feats > 0.75]

skewed_feats = skewed_feats.index



all_data[skewed_feats] = np.log1p(all_data[skewed_feats])
all_data.head()
all_data = all_data.fillna(all_data.mean())

all_data.head()
for name in all_data.columns:

    if (all_data[name].dtypes == 'object'):

        all_data[name].fillna(-1,inplace = True)



all_data.head()
le = preprocessing.LabelEncoder()

for name in all_data.columns:

    if (all_data[name].dtypes == 'object'):

        #all_data[name].fillna(')

        all_data[name] = le.fit_transform(all_data[name].astype(str))
all_data.head()
#creating matrices for sklearn:

X_train = all_data[:train.shape[0]]

X_test = all_data[train.shape[0]:]

y = train.SalePrice
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV, Lasso

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import train_test_split

import xgboost

from sklearn.ensemble import RandomForestRegressor

def rmse_cv(model):

    rmse= np.sqrt(-cross_val_score(model, X_train, y, scoring="neg_mean_squared_error", cv = 5))

    return(rmse)
x_train, x_valid, y_train, y_valid = train_test_split(X_train, y, test_size = 0.2, random_state = 1)
x_train
y_train
from xgboost import XGBRegressor

model = XGBRegressor()

model.fit(X=x_train,y=y_train,eval_set = [(x_train,y_train),(x_valid, y_valid)])
from sklearn.metrics import mean_squared_error

from math import sqrt



max_depths = [1,5,10,20,30,40,50,60,75, 80]

nums = [10, 20, 30, 40, 50, 60, 70,80,90,100]

for max_dept in max_depths:

    cv_randomforest = [rmse_cv(RandomForestRegressor(n_estimators = 20, max_depth = max_dept)).mean() for num in nums]

    cv_ridge = pd.Series(cv_randomforest, index = nums)

    cv_ridge.plot(title = "Validation - L")

    plt.xlabel("num")

    plt.ylabel("rmse")
cv_ridge = pd.Series(cv_randomforest, index = alphas)

cv_ridge.plot(title = "Validation - L")

plt.xlabel("alpha")

plt.ylabel("rmse")