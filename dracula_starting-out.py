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
train.tail()
matplotlib.rcParams['figure.figsize'] = (12.0, 6.0)

prices = pd.DataFrame({"price":train['SalePrice'],"log(price+1)":np.log1p(train['SalePrice'])})

prices.hist()
train["SalePrice"] = np.log1p(train["SalePrice"])
all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],

                      test.loc[:,'MSSubClass':'SaleCondition']))
numeric_feats = all_data.dtypes[all_data.dtypes != 'object'].index
skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna()))
all_data[skewed_feats.index] = np.log1p(all_data[skewed_feats.index])

all_data = pd.get_dummies(all_data)
all_data = all_data.fillna(all_data.mean())
X_train = all_data[:train.shape[0]]

X_test = all_data[train.shape[0]:]

y = train.SalePrice
from sklearn.linear_model import Lasso
ridlreg = Lasso(alpha=5e-4, max_iter=50000)
ridlreg.fit(X_train,y)
Y_predict = np.expm1(ridlreg.predict(X_test))
coef = pd.Series(ridlreg.coef_,index=X_train.columns);
imp_coef = pd.concat([coef.sort_values().head(10),coef.sort_values().tail(10)])
imp_coef.plot(kind='barh')
from sklearn.model_selection import cross_val_score

print(cross_val_score(ridlreg,X_train,y))
solution = pd.DataFrame({"id":test["Id"], "SalePrice":Y_predict})

solution.to_csv("ridge1.csv", index = False)
Y_predict[0:10]