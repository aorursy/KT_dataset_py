import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib

import matplotlib.pyplot as plt

from scipy.stats import skew



%config InlineBackend.figure_format = 'png' #set 'png' here when working on notebook

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
from sklearn.linear_model import LinearRegression, RANSACRegressor
ransac = RANSACRegressor(LinearRegression(),max_trials=100, min_samples=70,

                         residual_metric=lambda x: np.sum(np.abs(x), axis=1),

                         residual_threshold=2.0, random_state=1301)
ransac.fit(X_train, y)
#let's look at the residuals as well:

matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)



preds = pd.DataFrame({"preds":ransac.predict(X_train), "true":y})

preds["residuals"] = preds["true"] - preds["preds"]

preds.plot(x = "preds", y = "residuals",kind = "scatter")
rmse = np.sqrt(np.mean((preds['true']-preds['preds'])**2))

print ('RMSE: {0:.4f}'.format(rmse))
from sklearn.metrics import r2_score
print('R^2 train: %.3f' %  r2_score(preds['true'], preds['preds']))
coef = pd.Series(ransac.estimator_.coef_, index = X_train.columns)
imp_coef = pd.concat([coef.sort_values().head(10),

                     coef.sort_values().tail(10)])
matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)

imp_coef.plot(kind = "barh")

plt.title("Coefficients in the RANSAC Model")
preds = np.expm1(ransac.predict(X_test))
solution = pd.DataFrame({"id":test.Id, "SalePrice":preds})

solution.to_csv("ransac.csv", index = False)
from sklearn.linear_model import ElasticNet
elastic = ElasticNet(alpha=10.0, l1_ratio=0.005)
elastic.fit(X_train, y)
#let's look at the residuals as well:

matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)



preds = pd.DataFrame({"preds":elastic.predict(X_train), "true":y})

preds["residuals"] = preds["true"] - preds["preds"]

preds.plot(x = "preds", y = "residuals",kind = "scatter")
rmse = np.sqrt(np.mean((preds['true']-preds['preds'])**2))

print ('RMSE: {0:.4f}'.format(rmse))
print('R^2 train: %.3f' %  r2_score(preds['true'], preds['preds']))
from itertools import product
alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]

l1_ratios = [1, 0.1, 0.001, 0.0005]
from sklearn.model_selection import cross_val_score



def rmse_cv(model):

    rmse= np.sqrt(-cross_val_score(model, X_train, y, scoring="mean_squared_error", cv = 5))

    return(rmse)
cv_elastic = [rmse_cv(ElasticNet(alpha = alpha, l1_ratio=l1_ratio)).mean() 

            for (alpha, l1_ratio) in product(alphas, l1_ratios)]
matplotlib.rcParams['figure.figsize'] = (12.0, 6.0)

idx = list(product(alphas, l1_ratios))

p_cv_elastic = pd.Series(cv_elastic, index = idx)

p_cv_elastic.plot(title = "Validation - Just Do It")

plt.xlabel("alpha - l1_ratio")

plt.ylabel("rmse")
# Zoom in to the first 5 parameter pairs

matplotlib.rcParams['figure.figsize'] = (12.0, 6.0)

idx = list(product(alphas, l1_ratios))[:5]

p_cv_elastic = pd.Series(cv_elastic[:5], index = idx)

p_cv_elastic.plot(title = "Validation - Just Do It")

plt.xlabel("alpha - l1_ratio")

plt.ylabel("rmse")
elastic = ElasticNet(alpha=0.05, l1_ratio=0.001)
elastic.fit(X_train, y)
#let's look at the residuals as well:

matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)



preds = pd.DataFrame({"preds":elastic.predict(X_train), "true":y})

preds["residuals"] = preds["true"] - preds["preds"]

preds.plot(x = "preds", y = "residuals",kind = "scatter")
rmse = np.sqrt(np.mean((preds['true']-preds['preds'])**2))

print ('RMSE: {0:.4f}'.format(rmse))
print('R^2 train: %.3f' %  r2_score(preds['true'], preds['preds']))
coef = pd.Series(elastic.coef_, index = X_train.columns)
imp_coef = pd.concat([coef.sort_values().head(10),

                     coef.sort_values().tail(10)])
matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)

imp_coef.plot(kind = "barh")

plt.title("Coefficients in the Elastic Net Model")
preds = np.expm1(elastic.predict(X_test))
solution = pd.DataFrame({"id":test.Id, "SalePrice":preds})

solution.to_csv("elastic.csv", index = False)
matplotlib.rcParams['figure.figsize'] = (12.0, 6.0)

sns.regplot(x='PoolArea', y='SalePrice', data=train)
# How many houses have a pool

train[train.PoolArea > 0].shape
X_test['GrLivArea'].describe()
from sklearn.preprocessing import StandardScaler, RobustScaler