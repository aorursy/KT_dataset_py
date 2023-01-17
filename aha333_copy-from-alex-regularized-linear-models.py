from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"

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

test.head()

# Comments[Dawei]

## first of all there is 80 varialbes adding saleprice as the Y.

## Question is do we need to select variables ?

## several filters for feature selections(uni feature check, not related to Y) are : 

### 1 variance 

### 2 % missing values (Type I  missing at randoms records/

###                     Type II missing with pattern( in certain area))

### 
all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],

                      test.loc[:,'MSSubClass':'SaleCondition']))# this is just get rid of sale price
## Dawei: need to check more, at least missing value 

#all_data.isnull().sum()/all_data.count()

checkmissing=pd.DataFrame({'count':all_data.count(),'isnull':all_data.isnull().sum()})

checkmissing['missing%']=((checkmissing['isnull']/(checkmissing['count']+checkmissing['isnull']))*100).astype(int)

checkmissing[checkmissing['missing%']>10]
#all_data.variance()
matplotlib.rcParams['figure.figsize'] = (12.0, 6.0)

# this is very good way to check it 

prices = pd.DataFrame({"price":train["SalePrice"], "log(price + 1)":np.log1p(train["SalePrice"])})



prices.hist()
## Dawei Tricks

#train.dtypes.str.contains('str') this does not work, since .str.contains is for series of string type!!

# while df.dtypes is a series Not with element as string, but dtype

#  but does a string has this function??NO!! you can maybe use  ... in onestring, or string.find!

# method1 s.apply(lambda x: any(pd.Series(x).str.contains('you')))

# method2 s.apply(lambda x : any((i for i in x if i.find('you') >= 0)))

ifint=train.dtypes.apply( lambda x: str(x).find('int')>=0 )

list(train.dtypes[ifint].index)
#log transform the target:

train["SalePrice"] = np.log1p(train["SalePrice"])



#log transform skewed numeric features:

numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

### Dawei

### here the apply function would apply to each column!

## this is very good way: x.dropna() to get the non 

skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness

skewed_feats

skewed_feats = skewed_feats[skewed_feats > 0.75]

skewed_feats = skewed_feats.index



all_data[skewed_feats] = np.log1p(all_data[skewed_feats])    # all skewed are having log1 transform
# here is the problem!! what about missing value ??



#all_data.Alley

all_data = pd.get_dummies(all_data)
#filling NA's with the mean of the column:

all_data = all_data.fillna(all_data.mean())### this is really ---> I am not sure



## ## Dawei

## this missing value really need to understand!
#creating matrices for sklearn:

X_train = all_data[:train.shape[0]]

X_test = all_data[train.shape[0]:]

y = train.SalePrice
X_train.head(1)
#例子

import numpy as np

X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])

y = np.array([1, 1, 2, 2])

from sklearn.svm import SVC

SVC# you can see from blow this is model and its parameters( not fitted)

clf = SVC()

#from sklearn.exceptions import NotFittedError

#

#for model in models:

#   try:

#        model.predict(some_test_data)

#    except NotFittedError as e:

#        print(repr(e))



clf # ral classfier(a)

clf.fit(X, y) # Here is another model ( fitted)

clf.fit(X, y).score(X,y)## 这里非常重要， 每一个score都是相当入带入了新的data。

## 问题， 可以带test吗？？

## 对！！ svc.fit(X_train, y_train).score(X_test, y_test) 这个就是cross validation思路！！

#http://scikit-learn.org/stable/tutorial/statistical_inference/model_selection.html

# 这里就是python的办法

#[svc.fit(X_digits[train], y_digits[train]).score(X_digits[test], y_digits[test])

#...          for train, test in k_fold.split(X_digits)]



## 关于score http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter

## 其实每一个model是有自己默认的score， 但是呢， 如果用cross_val_score 的话， 会有更多的选择，。

## 大部分score其实就是用 y_fit and y_forecast 来判断的
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV

from sklearn.model_selection import cross_val_score



def rmse_cv(model):

    rmse= np.sqrt(-cross_val_score(model, X_train, y, scoring="neg_mean_squared_error", cv = 5))

    return(rmse)
#model_ridge = None#Ridge()
alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]

cv_ridge = [rmse_cv(Ridge(alpha = alpha)).mean() for alpha in alphas]

# here is the model with the run

#cv_ridge_scoreS =rmse_cv(Ridge(alpha = alphas))



cv_ridge = pd.Series(cv_ridge, index = alphas)

cv_ridge.plot(title = "Validation - Just Do It")

plt.xlabel("alpha")

plt.ylabel("rmse")
cv_ridge.min()

#cv_ridge.index(cv_ridge.min())

#cv_ridge.f

#aa=[1,2,3]

#aa.index(3)
## need to do the old way 

#alphas =  [1, 0.1, 0.001, 0.0005]





#cv_lasso = [rmse_cv(LassoCV(alpha = alpha)).mean() for alpha in alphas]

#LassoCV.alpha

model_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005]).fit(X_train, y)

model_lasso## 这里我不是很明白lasso——CV怎么实现的，CV 貌似默认是3folds
rmse_cv(model_lasso).mean()
coef = pd.Series(model_lasso.coef_, index = X_train.columns)

coef
print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")
imp_coef = pd.concat([coef.sort_values().head(10),

                     coef.sort_values().tail(10)])
matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)

imp_coef.plot(kind = "barh")

plt.title("Coefficients in the Lasso Model")
#let's look at the residuals as well:

matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)



preds = pd.DataFrame({"preds":model_lasso.predict(X_train), "true":y})

preds["residuals"] = preds["true"] - preds["preds"]

preds.plot(x = "preds", y = "residuals",kind = "scatter")
import xgboost as xgb


dtrain = xgb.DMatrix(X_train, label = y)

dtest = xgb.DMatrix(X_test)



params = {"max_depth":2, "eta":0.1}

model = xgb.cv(params, dtrain,  num_boost_round=500, early_stopping_rounds=100)

model

model.loc[30:,["test-rmse-mean", "train-rmse-mean"]].plot()
model_xgb = xgb.XGBRegressor(n_estimators=360, max_depth=2, learning_rate=0.1) 

#the params were tuned using xgb.cv

model_xgb.fit(X_train, y)
xgb_preds = np.expm1(model_xgb.predict(X_test))

lasso_preds = np.expm1(model_lasso.predict(X_test))
predictions = pd.DataFrame({"xgb":xgb_preds, "lasso":lasso_preds})

predictions.plot(x = "xgb", y = "lasso", kind = "scatter")
preds = 0.7*lasso_preds + 0.3*xgb_preds
solution = pd.DataFrame({"id":test.Id, "SalePrice":preds})

solution.to_csv("ridge_sol.csv", index = False)
from keras.layers import Dense

from keras.models import Sequential

from keras.regularizers import l1

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
X_train = StandardScaler().fit_transform(X_train)
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y, random_state = 3)
X_tr.shape
X_tr
model = Sequential()

#model.add(Dense(256, activation="relu", input_dim = X_train.shape[1]))

model.add(Dense(1, input_dim = X_train.shape[1], W_regularizer=l1(0.001)))



model.compile(loss = "mse", optimizer = "adam")
model.summary()
hist = model.fit(X_tr, y_tr, validation_data = (X_val, y_val))
pd.Series(model.predict(X_val)[:,0]).hist()