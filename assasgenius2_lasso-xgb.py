# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import seaborn as sns

import matplotlib



import matplotlib.pyplot as plt

from scipy.stats import skew

from scipy.stats.stats import pearsonr





%config InlineBackend.figure_format = 'retina' #set 'png' here when working on notebook

%matplotlib inline
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
train.head()
all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],

                      test.loc[:,'MSSubClass':'SaleCondition']))
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
all_data = pd.get_dummies(all_data)
#filling NA's with the mean of the column:

all_data = all_data.fillna(all_data.mean())
#creating matrices for sklearn:

X_train = all_data[:train.shape[0]]

X_test = all_data[train.shape[0]:]

y = train.SalePrice
from sklearn.linear_model import Ridge, RidgeCV, LassoCV, LassoLarsCV

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.model_selection import cross_val_score



def rmse_cv(model):

    rmse= np.sqrt(-cross_val_score(model, X_train, y, scoring="neg_mean_squared_error", cv = 3))

    return(rmse)
model_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005]).fit(X_train, y)
cv_rmse_lasso = rmse_cv(model_lasso).mean()
print (cv_rmse_lasso)
cv_lasso = pd.Series(model_lasso.coef_, index = X_train.columns)

cv_lasso.plot(title = "Validation Lasso")

plt.xlabel("coef")

plt.ylabel("rmse")
#let's look at the residuals as well:

matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)



preds_lasso= pd.DataFrame({"preds Lasso":model_lasso.predict(X_train), "true":y})

preds_lasso["residuals"] = preds_lasso["true"] - preds_lasso["preds Lasso"]

preds_lasso.plot(x = "preds Lasso", y = "residuals",kind = "scatter")
import xgboost as xgb

dtrain = xgb.DMatrix(X_train, label = y)

dtest = xgb.DMatrix(X_test)



params = {"max_depth":10, "eta":0.1}

model = xgb.cv(params, dtrain,  num_boost_round=500, early_stopping_rounds=100)
model.loc[30:,["test-rmse-mean", "train-rmse-mean"]].plot()
model_xgb = xgb.XGBRegressor(n_estimators=360, max_depth=10, learning_rate=0.1) #the params were tuned using xgb.cv

model_xgb.fit(X_train, y)
#gb_preds = np.expm1(model_gb.predict(X_test))

lasso_preds = np.expm1(model_lasso.predict(X_test))

#ridge_preds = np.expm1(model_ridge.predict(X_test))

xgb_preds = np.expm1(model_xgb.predict(X_test))
preds = 0.9*lasso_preds + 0.1*xgb_preds
solution = pd.DataFrame({"id":test.Id, "SalePrice":preds})

solution.to_csv("lasso_ridge_xgb_v2.csv", index = False)