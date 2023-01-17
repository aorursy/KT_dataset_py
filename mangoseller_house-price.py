

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

train.head()

train.shape
X_data = pd.concat((train.drop(['Id','SalePrice'],axis =1),

                      test.drop(['Id'],axis =1)))
missing = X_data.isnull().sum().sort_values(ascending=False)

missing = missing[missing > 0]

missing.plot.bar()

missing.head(6)
missing.head(6)
corrmat = train[['PoolQC','MiscFeature','Alley','Fence','FireplaceQu','LotFrontage','SalePrice']].corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10})
X_data = X_data.fillna(X_data.mean())

X_data = X_data.fillna("None")
missing = X_data.isnull().sum().sort_values(ascending=False)

missing = missing[missing > 0]

missing.head()
train2 = train.fillna(X_data.mean())

train2 = train.fillna("None")

train2.shape
#correlation matrix

corrmat = train2.corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, vmax=.75);
#correlation matrix

corrmat = train2.corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, vmax=.75,vmin =.5);
sns.distplot(train['SalePrice'],bins = 15);
from scipy.stats import skew

test1 = skew(train['SalePrice'])

test1
sns.distplot(np.log1p(train['SalePrice']),bins = 15);

train["SalePrice"] = np.log1p(train["SalePrice"])

test2 = skew(train['SalePrice'])

test2
numeric_feats = X_data.dtypes[X_data.dtypes != "object"].index



skewed_feats = train[numeric_feats].apply(lambda x: skew(x)) #compute skewness

skewed_feats = skewed_feats[skewed_feats > 0.75]

skewed_feats = skewed_feats.index



X_data[skewed_feats] = np.log1p(X_data[skewed_feats])
# one hot encoding and test/train split

X_data = pd.get_dummies(X_data)



X_train = X_data[:train.shape[0]]

X_test = X_data[train.shape[0]:]

y = train.SalePrice
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV

from sklearn.model_selection import cross_val_score



def rmse_cv(model):

    rmse= np.sqrt(-cross_val_score(model, X_train, y, scoring="neg_mean_squared_error", cv = 5))

    return(rmse)
model_ridge = Ridge()
model_ridge = Ridge()

alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]

cv_ridge = [rmse_cv(Ridge(alpha = alpha)).mean() 

            for alpha in alphas]
cv_ridge = pd.Series(cv_ridge, index = alphas)

cv_ridge.plot(title = "Validation")

plt.xlabel("alpha")

plt.ylabel("rmse")
cv_ridge.min()
model_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005]).fit(X_train, y)
rmse_cv(model_lasso).mean()
coef = pd.Series(model_lasso.coef_, index = X_train.columns)

print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")
imp_coef = pd.concat([coef.sort_values().head(10),

                     coef.sort_values().tail(10)])

plt.rcParams['figure.figsize'] = (8.0, 10.0)

imp_coef.plot(kind = "barh")

plt.title("Coefficients in the Lasso Model")
# residuals

plt.rcParams['figure.figsize'] = (6.0, 6.0)



preds = pd.DataFrame({"preds":model_lasso.predict(X_train), "true":y})

preds["residuals"] = preds["true"] - preds["preds"]

preds.plot(x = "preds", y = "residuals",kind = "scatter")
lasso_preds = np.expm1(model_lasso.predict(X_test))

solution = pd.DataFrame({"id": test['Id'], "SalePrice": lasso_preds})

solution.to_csv("lasso_sol.csv", index = False)