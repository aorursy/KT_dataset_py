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
from sklearn.cross_validation import cross_val_score

from sklearn.metrics import make_scorer, mean_squared_error



scorer = make_scorer(mean_squared_error, False)



def rmse_cv(model, X, y):

     return (cross_val_score(model, X, y, scoring=scorer)).mean()
from sklearn.linear_model import Lasso

alphas = [1e-4, 5e-4, 1e-3, 5e-3]

cv_lasso = [rmse_cv(Lasso(alpha = alpha, max_iter = 50000), X_train, y) for alpha in alphas]

pd.Series(cv_lasso, index = alphas).plot()
model_lasso = Lasso(alpha = 5e-4, max_iter=50000).fit(X_train, y)
coef = pd.Series(model_lasso.coef_, index = X_train.columns).sort_values()

imp_coef = pd.concat([coef.head(10), coef.tail(10)])

imp_coef.plot(kind = "barh")

plt.title("Coefficients in the Model")
# This is a good way to see how model predict data

pred_train = np.expm1(model_lasso.predict(X_train))

plt.scatter(pred_train, np.expm1(y))

plt.plot([min(pred_train),max(pred_train)], [min(pred_train),max(pred_train)], c="red")
lasso_preds = np.expm1(model_lasso.predict(X_test))
preds = lasso_preds
solution = pd.DataFrame({"id":test.Id, "SalePrice":preds})

solution.to_csv("lasso_sol.csv", index = False)