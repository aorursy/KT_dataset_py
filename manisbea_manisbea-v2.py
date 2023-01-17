import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib



import matplotlib.pyplot as plt

from scipy.stats import skew

from scipy.stats.stats import pearsonr





%config InlineBackend.figure_format = 'retina' #set 'png' here when working on notebook

%matplotlib inline
train = '../input/train.csv'

test = '../input/test.csv'



train = pd.read_csv(train)

test = pd.read_csv(test)


train.head()


all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],

                      test.loc[:,'MSSubClass':'SaleCondition']))
matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)

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
from sklearn.linear_model import Ridge, RidgeCV, LassoCV, LassoLarsCV,ElasticNet, BayesianRidge

from sklearn.model_selection import cross_val_score

from sklearn.ensemble import GradientBoostingRegressor



def rmse_cv(model):

    rmse= np.sqrt(-cross_val_score(model, X_train, y, scoring="neg_mean_squared_error", cv = 5))

    return(rmse)
model_ridge = Ridge()
alphas = [0.05, 0.1, 0.3,0.5,0.7, 1, 3, 5,10,15,22, 30, 50,70,80,100]

cv_ridge = [rmse_cv(Ridge(alpha = alpha)).mean() 

            for alpha in alphas]
cv_ridge = pd.Series(cv_ridge, index = alphas)

cv_ridge.plot(title = "Validation - Ridge")

plt.xlabel("alpha")

plt.ylabel("rmse")
cv_ridge.min()
model_ridge = Ridge(alpha = 15).fit(X_train, y)
matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)



preds_ridge = pd.DataFrame({"preds":model_ridge.predict(X_train), "true":y})

preds_ridge["residuals"] = preds_ridge["true"] - preds_ridge["preds"]

preds_ridge.plot(x = "preds", y = "residuals",kind = "scatter")
model_BayesianRidge = BayesianRidge()
cv_Bayridge = [rmse_cv(BayesianRidge(alpha_1 =  1e-20, alpha_2 = 1e-20, lambda_1 = 1e-20, lambda_2= 1e-20)).mean()]
#cv_Bayridge.min()
model_BayesianRidge = BayesianRidge(alpha_1 = 1e-06).fit(X_train, y)
matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)



preds_Bayridge = pd.DataFrame({"preds":model_BayesianRidge.predict(X_train), "true":y})

preds_Bayridge["residuals"] = preds_Bayridge["true"] - preds_Bayridge["preds"]

preds_Bayridge.plot(x = "preds", y = "residuals",kind = "scatter")
model_lasso = LassoCV(alphas = [1,0.5, 0.1, 0.001, 0.0005]).fit(X_train, y)
rmse_lasso=rmse_cv(model_lasso).mean()
print (rmse_lasso)
cv_lasso = pd.Series(model_lasso.coef_, index = X_train.columns)

cv_lasso.plot(title = "Validation Lasso")

plt.xlabel("coef")

plt.ylabel("rmse")
coef = pd.Series(model_lasso.coef_, index = X_train.columns)
print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")
imp_coef = pd.concat([coef.sort_values().head(10),

                     coef.sort_values().tail(10)])
matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)

imp_coef.plot(kind = "barh")

plt.title("Coefficients in the Lasso Model")
#let's look at the residuals as well:

matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)



preds_lasso = pd.DataFrame({"preds":model_lasso.predict(X_train), "true":y})

preds_lasso["residuals"] = preds_lasso["true"] - preds_lasso["preds"]

preds_lasso.plot(x = "preds", y = "residuals",kind = "scatter")
model_elasticnet = ElasticNet ()

alphas_en = [0.001, 0.005, 0.1, 0.2, 0.3]

cv_rmse_en = [rmse_cv(ElasticNet(alpha = alpha)).mean() 

            for alpha in alphas_en]
print (cv_rmse_en)
cv_en = pd.Series(cv_rmse_en, index = alphas_en)

cv_en.plot(title = "Validation - Elastic Net")

plt.xlabel("alphas")

plt.ylabel("rmse")
cv_en.min()
model_elasticnet = ElasticNet(alpha = 0.001).fit(X_train, y)
#let's look at the residuals as well:

matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)



preds_en = pd.DataFrame({"preds EN":model_elasticnet.predict(X_train), "true":y})

preds_en["residuals"] = preds_en["true"] - preds_en["preds EN"]

preds_en.plot(x = "preds EN", y = "residuals",kind = "scatter")
n_estimators = [150, 200 ,300, 400, 500, 600]

cv_rmse_gb = [rmse_cv(GradientBoostingRegressor(n_estimators = n_estimator)).mean() 

            for n_estimator in n_estimators]
print (cv_rmse_gb)
cv_gb = pd.Series(cv_rmse_gb , index = n_estimators)

cv_gb.plot(title = "Validation Gradient Boosting")

plt.xlabel("n_estimator")

plt.ylabel("rmse")
cv_gb.min()
model_gb = GradientBoostingRegressor(n_estimators = 500).fit(X_train, y)
#let's look at the residuals as well:

matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)



preds_gb = pd.DataFrame({"preds Gradient Boost":model_gb.predict(X_train), "true":y})

preds_gb["residuals"] = preds_gb["true"] - preds_gb["preds Gradient Boost"]

preds_gb.plot(x = "preds Gradient Boost", y = "residuals",kind = "scatter")
import xgboost as xgb
dtrain = xgb.DMatrix(X_train, label = y)

dtest = xgb.DMatrix(X_test)



params = {"max_depth":10, "eta":0.1}

model = xgb.cv(params, dtrain,  num_boost_round=500, early_stopping_rounds=100)
model.loc[30:,["test-rmse-mean", "train-rmse-mean"]].plot()
model_xgb = xgb.XGBRegressor(n_estimators=360, max_depth=2, learning_rate=0.1) #the params were tuned using xgb.cv

model_xgb.fit(X_train, y)


#preds_ridge = np.expm1(model_ridge.predict(X_test))

preds_xgb = np.expm1(model_xgb.predict(X_test))

preds_gb = np.expm1(model_gb.predict(X_test))

preds_lasso= np.expm1(model_lasso.predict(X_test))

preds_en=np.expm1(model_elasticnet.predict(X_test))
preds = 0.65*preds_lasso+0.15*preds_xgb+0.2*preds_en
solution = pd.DataFrame({"id":test.Id, "SalePrice":preds})

solution.to_csv("ridge_sol.csv", index = False)