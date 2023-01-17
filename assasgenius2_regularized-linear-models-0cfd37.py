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
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV

from sklearn.model_selection import cross_val_score



def rmse_cv(model):

    rmse= np.sqrt(-cross_val_score(model, X_train, y, scoring="neg_mean_squared_error", cv = 5))

    return(rmse)
model_ridge = Ridge()
from sklearn.ensemble import RandomForestRegressor

n_estimators = [10 , 20 , 50 , 170]

cv_rmse_rf = [rmse_cv(RandomForestRegressor(n_estimators = n_estimator)).mean() 

            for n_estimator in n_estimators]
cv_rf = pd.Series(cv_rmse_rf , index = n_estimators)

cv_rf.plot(title = "Validation - Just Do It")

plt.xlabel("n_estimator")

plt.ylabel("rmse")
print("RF picked " + str(sum(cv_rf != 0)) + " variables and eliminated the other " +  str(sum(cv_rf == 0)) + " variables")
cv_rf.min()
model_rf = RandomForestRegressor(n_estimators = 150).fit(X_train, y)
model_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005]).fit(X_train, y)
rmse_cv(model_lasso).mean()
coef = pd.Series(model_lasso.coef_)
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

preds.plot(x = "preds", y = "residuals",kind = "scatter")
model_preds = np.expm1(model_rf.predict(X_test))

lasso_preds = np.expm1(model_lasso.predict(X_test))
preds_rf = pd.DataFrame({"preds":model_rf.predict(X_train), "true":y})

preds_rf["residuals"] = preds_rf["true"] - preds_rf["preds"]

preds.plot(x = "preds", y = "residuals",kind = "scatter")

predictions.plot(x = "xgb", y = "lasso", kind = "scatter")
preds = 0.7*lasso_preds + 0.3*randomforest_preds
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