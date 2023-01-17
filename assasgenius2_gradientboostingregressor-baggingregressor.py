import pandas as pd

import numpy as np

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
from sklearn.linear_model import LassoCV, LassoLarsCV

from sklearn.model_selection import cross_val_score



def rmse_cv(model):

    rmse= np.sqrt(-cross_val_score(model, X_train, y, scoring="neg_mean_squared_error", cv = 3))

    return(rmse)
from sklearn.ensemble import GradientBoostingRegressor

n_estimators = [150, 170 , 200, 400, 500]

cv_rmse_gb = [rmse_cv(GradientBoostingRegressor(n_estimators = n_estimator)).mean() 

            for n_estimator in n_estimators]
print (cv_rmse_gb)
cv_gb = pd.Series(cv_rmse_gb , index = n_estimators)

cv_gb.plot(title = "Validation Gradient Boosting")

plt.xlabel("n_estimator")

plt.ylabel("rmse")
cv_gb.min()
model_gb = GradientBoostingRegressor(n_estimators = 400).fit(X_train, y)
#let's look at the residuals as well:

matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)



preds_gb = pd.DataFrame({"preds Gradient Boost":model_gb.predict(X_train), "true":y})

preds_gb["residuals"] = preds_gb["true"] - preds_gb["preds Gradient Boost"]

preds_gb.plot(x = "preds Gradient Boost", y = "residuals",kind = "scatter")
from sklearn.ensemble import BaggingRegressor

n_estimators = [170, 200, 250, 300]

cv_rmse_br = [rmse_cv(BaggingRegressor(n_estimators = n_estimator)).mean() 

            for n_estimator in n_estimators]
print (cv_rmse_br)
cv_br = pd.Series(cv_rmse_br , index = n_estimators)

cv_br.plot(title = "Validation BaggingRegressor")

plt.xlabel("n_estimator")

plt.ylabel("rmse")
cv_br.min()
model_br = BaggingRegressor(n_estimators = 250).fit(X_train, y)
br_preds = np.expm1(model_br.predict(X_test))

gb_preds = np.expm1(model_gb.predict(X_test))
preds = 0.5*br_preds + 0.5*gb_preds
solution = pd.DataFrame({"id":test.Id, "SalePrice":preds})

solution.to_csv("gb_br_sol.csv", index = False)