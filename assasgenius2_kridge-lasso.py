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
from sklearn.linear_model import LassoCV, LassoLarsCV, Lasso

from sklearn.model_selection import cross_val_score



def rmse_cv(model):

    rmse= np.sqrt(-cross_val_score(model, X_train, y, scoring="neg_mean_squared_error", cv = 5))

    return(rmse)
alphas_lasso = [0.1, 0.001, 0.0005]
cv_rmse_lasso = [rmse_cv(Lasso(alpha = alpha)).mean()

                 for alpha in alphas_lasso]
print (cv_rmse_lasso)
cv_lasso = pd.Series(cv_rmse_lasso , index = alphas_lasso)

cv_lasso.plot(title = "Validation Lasso")

plt.xlabel("alphas")

plt.ylabel("rmse")
cv_lasso.min()
model_lasso= Lasso(alpha = 0.0005).fit(X_train, y)
#let's look at the residuals as well:

matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)



preds_lasso= pd.DataFrame({"preds Lasso":model_lasso.predict(X_train), "true":y})

preds_lasso["residuals"] = preds_lasso["true"] - preds_lasso["preds Lasso"]

preds_lasso.plot(x = "preds Lasso", y = "residuals",kind = "scatter")
from sklearn.kernel_ridge import KernelRidge

alphas_kridge = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]

cv_rmse_kridge = [rmse_cv(KernelRidge(alpha = alpha)).mean() 

            for alpha in alphas_kridge]
print (cv_rmse_kridge)
cv_kridge = pd.Series(cv_rmse_kridge , index = alphas_kridge)

cv_kridge.plot(title = "Validation Kernel Ridge")

plt.xlabel("alphas")

plt.ylabel("rmse")
cv_kridge.min()
model_kridge= KernelRidge(alpha = 10).fit(X_train, y)
#let's look at the residuals as well:

matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)



preds_kridge = pd.DataFrame({"preds KRidge":model_kridge.predict(X_train), "true":y})

preds_kridge["residuals"] = preds_kridge["true"] - preds_kridge["preds KRidge"]

preds_kridge.plot(x = "preds KRidge", y = "residuals",kind = "scatter")
kridge_preds = np.expm1(model_kridge.predict(X_test))

lasso_preds = np.expm1(model_lasso.predict(X_test))
preds = 0.5*lasso_preds + 0.5*kridge_preds
print (preds)
solution = pd.DataFrame({"id":test.Id, "SalePrice":preds})

solution.to_csv("kridge_lasso_sol.csv", index = False)