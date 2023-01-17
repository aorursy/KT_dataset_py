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
all_data.shape
X_train.shape
X_test.shape
y.shape
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV

from sklearn.model_selection import cross_val_score



# define a function that returns the cross-validation rmse error 

# so we can evaluate our models and pick the best tuning par

def rmse_cv(model):

    rmse= np.sqrt(-cross_val_score(model, X_train, y, scoring="neg_mean_squared_error", cv = 5))

    return(rmse)
alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]

cv_ridge = [rmse_cv(Ridge(alpha = alpha)).mean() 

            for alpha in alphas]
cv_ridge = pd.Series(cv_ridge, index = alphas)

cv_ridge.plot(title = "Validation - Ridge Model")

plt.xlabel("alpha")

plt.ylabel("rmse")
cv_ridge.min()
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
from sklearn.ensemble import RandomForestRegressor
n_estimators = [10, 20, 50, 100]

cv_rf1 = [rmse_cv(RandomForestRegressor(n_estimators = n_estimator, random_state=1)).mean() 

            for n_estimator in n_estimators]

cv_rf1 = pd.Series(cv_rf1, index = n_estimators)

cv_rf1.plot(title = "Validation - Random Forest Regression")

plt.xlabel("n_estimators")

plt.ylabel("rmse")
cv_rf1.min()
max_depths = [5, 10, 15, 20, 25]

cv_rf2 = [rmse_cv(RandomForestRegressor(n_estimators = 100, max_depth = max_depth,  random_state=1)).mean() 

            for max_depth in max_depths]

cv_rf2 = pd.Series(cv_rf2, index = max_depths)

cv_rf2.plot(title = "Validation - Random Forest Regression")

plt.xlabel("max_depth")

plt.ylabel("rmse")
cv_rf2.min()
model_rf = RandomForestRegressor(n_estimators = 100, max_depth = 15, random_state =1)

model_rf.fit(X_train,y)

preds_rf =np.expm1( model_rf.predict(X_test))

preds_rf
from sklearn.ensemble import GradientBoostingRegressor
n_estimators = [200, 400, 500, 600]

cv_gb1 = [rmse_cv(GradientBoostingRegressor(n_estimators = n_estimator, random_state=1)).mean() 

            for n_estimator in n_estimators]



cv_gb1 = pd.Series(cv_gb1 , index = n_estimators)

cv_gb1.plot(title = "Validation - Gradient Boosting")

plt.xlabel("n_estimator")

plt.ylabel("rmse")
cv_gb1.min()
max_depths = [1, 2, 3, 4, 5]

cv_gb2 = [rmse_cv(GradientBoostingRegressor(n_estimators = 400, max_depth = max_depth,  random_state=1)).mean() 

            for max_depth in max_depths]

cv_gb2 = pd.Series(cv_gb2, index = max_depths)

cv_gb2.plot(title = "Validation - Gradient Boosting")

plt.xlabel("max_depth")

plt.ylabel("rmse")
cv_gb2.min()
model_gb = GradientBoostingRegressor(n_estimators = 400, max_depth = 3,random_state=1).fit(X_train, y)

model_gb.fit(X_train,y)
from sklearn.ensemble import BaggingRegressor

n_estimators = [400, 500, 600]

cv_br = [rmse_cv(BaggingRegressor(n_estimators = n_estimator, random_state=1)).mean() 

            for n_estimator in n_estimators]
cv_br = pd.Series(cv_br , index = n_estimators)

cv_br.plot(title = "Validation - Bagging Regressor")

plt.xlabel("n_estimator")

plt.ylabel("rmse")
cv_br.min()
model_br = BaggingRegressor(n_estimators = 500).fit(X_train, y)
from sklearn.kernel_ridge import KernelRidge

alphas_kr = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]

cv_kridge = [rmse_cv(KernelRidge(alpha = alpha)).mean() 

            for alpha in alphas_kr]
cv_kridge = pd.Series(cv_kridge , index = alphas_kr)

cv_kridge.plot(title = "Validation - Kernel Ridge")

plt.xlabel("alphas")

plt.ylabel("rmse")
cv_kridge.min()
model_kridge= KernelRidge(alpha = 10).fit(X_train, y)
import xgboost as xgb



dtrain = xgb.DMatrix(X_train, label = y)

dtest = xgb.DMatrix(X_test)



params = {"max_depth":2, "eta":0.1}

model = xgb.cv(params, dtrain,  num_boost_round=500, early_stopping_rounds=100)
cv_xgb = rmse_cv(xgb.XGBRegressor(n_estimators=360, max_depth=2, learning_rate=0.1)).mean() 
cv_xgb
model.loc[30:,["test-rmse-mean", "train-rmse-mean"]].plot()
model_xgb = xgb.XGBRegressor(n_estimators=360, max_depth=2, learning_rate=0.1) #the params were tuned using xgb.cv

model_xgb.fit(X_train, y)
gb_preds = np.expm1(model_gb.predict(X_test))

lasso_preds = np.expm1(model_lasso.predict(X_test))

kridge_preds = np.expm1(model_kridge.predict(X_test))
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt

import statsmodels.formula.api as sm

from matplotlib import cm

predictions = pd.DataFrame({"kridge":kridge_preds, "lasso":lasso_preds, "gb": gb_preds})

fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')



x_surf = np.arange(0, 350, 20)                # generate a mesh

y_surf = np.arange(0, 60, 4)

x_surf, y_surf = np.meshgrid(x_surf, y_surf)



ax.plot_surface(kridge_preds,lasso_preds,

                gb_preds.reshape(kridge_preds.shape),

                rstride=1,

                cstride=1,

                color='None',

                alpha = 0.4)



ax.scatter(predictions['kridge'], predictions['lasso'], predictions['gb'],

           c='red',

           marker='.',

           alpha=1)





ax.set_xlabel('kridge')

ax.set_ylabel('lasso')

ax.set_zlabel('gb')
preds = 0.5*gb_preds + 0.3*lasso_preds + 0.2*kridge_preds

preds
from sklearn.model_selection import KFold

kf = KFold(n_splits=2)

kf.get_n_splits(X_train)
print(kf)
for train_index, test_index in kf.split(X_train):

    print("TRAIN:", train_index, "TEST:", test_index)

solution = pd.DataFrame({"id":test.Id, "SalePrice":preds})

solution.to_csv("House_Price.csv", index = False)