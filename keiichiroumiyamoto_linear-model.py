import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib



import matplotlib.pyplot as plt

from sklearn import svm



from scipy.stats import skew

from scipy.stats.stats import pearsonr



from itertools import cycle

import numpy as np

import matplotlib.pyplot as plt



from sklearn.linear_model import lasso_path, enet_path

from sklearn.ensemble import GradientBoostingRegressor 

from sklearn.grid_search import GridSearchCV







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

print(X_train)
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV

from sklearn.model_selection import cross_val_score



def rmse_cv(model):

    rmse= np.sqrt(-cross_val_score(model, X_train, y, scoring="neg_mean_squared_error", cv = 5))

    return(rmse)
model_ridge = Ridge()
alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]

cv_ridge = [rmse_cv(Ridge(alpha = alpha)).mean() 

            for alpha in alphas]
cv_ridge = pd.Series(cv_ridge, index = alphas)

cv_ridge.plot(title = "Validation - Just Do It")

plt.xlabel("alpha")

plt.ylabel("rmse")
cv_ridge.min()
model_ridge = Ridge(alpha = 5).fit(X_train, y)
#let's look at the residuals as well:

matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)



preds_ridge = pd.DataFrame({"preds Ridge":model_ridge.predict(X_train), "true":y})

preds_ridge["residuals"] = preds_ridge["true"] - preds_ridge["preds Ridge"]

preds_ridge.plot(x = "preds Ridge", y = "residuals",kind = "scatter")
model_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005]).fit(X_train, y)
rmse_cv(model_lasso).mean()
coef = pd.Series(model_lasso.coef_, index = X_train.columns)
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
model_elas = ElasticNet(alpha=0.001, l1_ratio=0.5, fit_intercept=True, normalize=False, precompute=False, max_iter=1000, copy_X=True, tol=0.0001, warm_start=False, positive=False, random_state=None, selection='cyclic').fit(X_train, y)
rmse_cv(model_elas).mean()
#let's look at the residuals as well:

matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)



preds_elas = pd.DataFrame({"preds Elas":model_elas.predict(X_train), "true":y})

preds_elas["residuals"] = preds_elas["true"] - preds_elas["preds Elas"]

preds_elas.plot(x = "preds Elas", y = "residuals",kind = "scatter")
import xgboost as xgb


dtrain = xgb.DMatrix(X_train, label = y)

dtest = xgb.DMatrix(X_test)



params = {"max_depth":6, "eta":0.1}

model = xgb.cv(params, dtrain,  num_boost_round=500, early_stopping_rounds=100)
model.loc[30:,["test-rmse-mean", "train-rmse-mean"]].plot()
model_xgb = xgb.XGBRegressor(n_estimators=360, max_depth=6, learning_rate=0.1) #the params were tuned using xgb.cv

model_xgb.fit(X_train, y)
rmse_cv(model_xgb).mean()
#let's look at the residuals as well:

matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)



preds_elas = pd.DataFrame({"model_xgb":model_elas.predict(X_train), "true":y})

preds_elas["residuals"] = preds_elas["true"] - preds_elas["model_xgb"]

preds_elas.plot(x = "model_xgb", y = "residuals",kind = "scatter")
model_gd = GradientBoostingRegressor(loss='ls', learning_rate=0.1, n_estimators=100, subsample=1.0, criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_split=1e-07, init=None, random_state=None, max_features=None, alpha=0.09, verbose=0, max_leaf_nodes=None, warm_start=False, presort='auto')

model_gd.fit(X_train, y)
rmse_cv(model_gd).mean()
#let's look at the residuals as well:

matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)



preds_elas = pd.DataFrame({"model_gd":model_elas.predict(X_train), "true":y})

preds_elas["residuals"] = preds_elas["true"] - preds_elas["model_gd"]

preds_elas.plot(x = "model_gd", y = "residuals",kind = "scatter")
#gb_preds = np.expm1(model_gb.predict(X_test))

lasso_preds = np.expm1(model_lasso.predict(X_test))

ridge_preds = np.expm1(model_ridge.predict(X_test))

xgb_preds = np.expm1(model_xgb.predict(X_test))

gd_preds = np.expm1(model_gd.predict(X_test))

elas_preds = np.expm1(model_elas.predict(X_test))
predictions = pd.DataFrame({"xgb":xgb_preds, "lasso":lasso_preds})

predictions.plot(x = "xgb", y = "lasso", kind = "scatter")
predictions = pd.DataFrame({"gd":gd_preds, "lasso":lasso_preds})

predictions.plot(x = "gd", y = "lasso", kind = "scatter")
predictions = pd.DataFrame({"gd":gd_preds, "xgb":xgb_preds})

predictions.plot(x = "gd", y = "xgb", kind = "scatter")
prediction = 0.65*lasso_preds + 0.15*ridge_preds + 0.20*xgb_preds
solution = pd.DataFrame({"id":test.Id, "SalePrice":preds})

solution.to_csv("ridge_sol.csv", index = False)
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib



import matplotlib.pyplot as plt

from sklearn import svm



from scipy.stats import skew

from scipy.stats.stats import pearsonr



from itertools import cycle

import numpy as np

import matplotlib.pyplot as plt



from sklearn.linear_model import lasso_path, enet_path

from sklearn.ensemble import GradientBoostingRegressor 

from sklearn.grid_search import GridSearchCV







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

print(X_train)
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV

from sklearn.model_selection import cross_val_score



def rmse_cv(model):

    rmse= np.sqrt(-cross_val_score(model, X_train, y, scoring="neg_mean_squared_error", cv = 5))

    return(rmse)
model_ridge = Ridge()
alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]

cv_ridge = [rmse_cv(Ridge(alpha = alpha)).mean() 

            for alpha in alphas]
cv_ridge = pd.Series(cv_ridge, index = alphas)

cv_ridge.plot(title = "Validation - Just Do It")

plt.xlabel("alpha")

plt.ylabel("rmse")
cv_ridge.min()
model_ridge = Ridge(alpha = 5).fit(X_train, y)
#let's look at the residuals as well:

matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)



preds_ridge = pd.DataFrame({"preds Ridge":model_ridge.predict(X_train), "true":y})

preds_ridge["residuals"] = preds_ridge["true"] - preds_ridge["preds Ridge"]

preds_ridge.plot(x = "preds Ridge", y = "residuals",kind = "scatter")
model_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005]).fit(X_train, y)
rmse_cv(model_lasso).mean()
coef = pd.Series(model_lasso.coef_, index = X_train.columns)
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
model_elas = ElasticNet(alpha=0.001, l1_ratio=0.5, fit_intercept=True, normalize=False, precompute=False, max_iter=1000, copy_X=True, tol=0.0001, warm_start=False, positive=False, random_state=None, selection='cyclic').fit(X_train, y)
rmse_cv(model_elas).mean()
#let's look at the residuals as well:

matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)



preds_elas = pd.DataFrame({"preds Elas":model_elas.predict(X_train), "true":y})

preds_elas["residuals"] = preds_elas["true"] - preds_elas["preds Elas"]

preds_elas.plot(x = "preds Elas", y = "residuals",kind = "scatter")
import xgboost as xgb


dtrain = xgb.DMatrix(X_train, label = y)

dtest = xgb.DMatrix(X_test)



params = {"max_depth":6, "eta":0.1}

model = xgb.cv(params, dtrain,  num_boost_round=500, early_stopping_rounds=100)
model.loc[30:,["test-rmse-mean", "train-rmse-mean"]].plot()
model_xgb = xgb.XGBRegressor(n_estimators=360, max_depth=6, learning_rate=0.1) #the params were tuned using xgb.cv

model_xgb.fit(X_train, y)
rmse_cv(model_xgb).mean()
#let's look at the residuals as well:

matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)



preds_elas = pd.DataFrame({"model_xgb":model_elas.predict(X_train), "true":y})

preds_elas["residuals"] = preds_elas["true"] - preds_elas["model_xgb"]

preds_elas.plot(x = "model_xgb", y = "residuals",kind = "scatter")
model_gd = GradientBoostingRegressor(loss='ls', learning_rate=0.1, n_estimators=100, subsample=1.0, criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_split=1e-07, init=None, random_state=None, max_features=None, alpha=0.09, verbose=0, max_leaf_nodes=None, warm_start=False, presort='auto')

model_gd.fit(X_train, y)
rmse_cv(model_gd).mean()
#let's look at the residuals as well:

matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)



preds_elas = pd.DataFrame({"model_gd":model_elas.predict(X_train), "true":y})

preds_elas["residuals"] = preds_elas["true"] - preds_elas["model_gd"]

preds_elas.plot(x = "model_gd", y = "residuals",kind = "scatter")
#gb_preds = np.expm1(model_gb.predict(X_test))

lasso_preds = np.expm1(model_lasso.predict(X_test))

ridge_preds = np.expm1(model_ridge.predict(X_test))

xgb_preds = np.expm1(model_xgb.predict(X_test))

gd_preds = np.expm1(model_gd.predict(X_test))

elas_preds = np.expm1(model_elas.predict(X_test))
predictions = pd.DataFrame({"xgb":xgb_preds, "lasso":lasso_preds})

predictions.plot(x = "xgb", y = "lasso", kind = "scatter")
predictions = pd.DataFrame({"gd":gd_preds, "lasso":lasso_preds})

predictions.plot(x = "gd", y = "lasso", kind = "scatter")
predictions = pd.DataFrame({"gd":gd_preds, "xgb":xgb_preds})

predictions.plot(x = "gd", y = "xgb", kind = "scatter")
prediction = 0.65*lasso_preds + 0.15*ridge_preds + 0.20*xgb_preds
solution = pd.DataFrame({"id":test.Id, "SalePrice":preds})

solution.to_csv("ridge_sol.csv", index = False)