# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib

import matplotlib.pyplot as plt





%config InlineBackend.figure_format = 'png' #set 'png' here when working on notebook

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
train.info()
import seaborn as sns
train.plot.scatter(x='OverallQual', y='SalePrice');
# Relation is not linear

sns.lmplot(x = 'OverallQual', y = 'SalePrice', data = train);
# Let's try a second order polynomial

sns.lmplot(x = 'OverallQual', y = 'SalePrice', data = train, order=2, x_estimator=np.median);
# Hm, looks better but SalePrice goes up when OverallQual gets below 2

# Lets try a third order

sns.lmplot(x = 'OverallQual', y = 'SalePrice', data = train, order=3);
# Okay, lets stick with a third order polynomial

p = np.polyfit(train['OverallQual'], train['SalePrice'], 3)
p
rmse = np.sqrt(np.square( train['SalePrice']-np.polyval(p,train['OverallQual']) ).mean())

print ('Root mean square error: {:.3f}'.format(rmse))
prediction = np.polyval(p, test['OverallQual'])
solution = pd.DataFrame({"id":test.Id, "SalePrice":prediction})

solution.to_csv("third_order_poly.csv", index = False)
# This scored already a 0.23273 at the leaderboard
all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],

                      test.loc[:,'MSSubClass':'SaleCondition']))
train["diff_SalePrice"] = train["SalePrice"]-np.polyval(p, train['OverallQual'])
train["diff_SalePrice"].hist(bins=100);
from scipy.stats import skew



#log transform skewed numeric features:

numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index



skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness

skewed_feats = skewed_feats[skewed_feats > 0.75]

skewed_feats = skewed_feats.index



all_data[skewed_feats] = np.log1p(all_data[skewed_feats])
#Check if the fitted polynomial can be used

'OverallQual' in skewed_feats
all_data = pd.get_dummies(all_data)

#filling NA's with the mean of the column:

all_data = all_data.fillna(all_data.mean())
#creating matrices for sklearn:

X_train = all_data[:train.shape[0]]

X_test = all_data[train.shape[0]:]

y = train.diff_SalePrice # We now use the difference as target variabele
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV

from sklearn.model_selection import cross_val_score



def rmse_cv(model):

    rmse= np.sqrt(-cross_val_score(model, X_train, y, scoring="neg_mean_squared_error", cv = 5))

    return(rmse)
model_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005]).fit(X_train, y)
rmse_cv(model_lasso).mean()
coef = pd.Series(model_lasso.coef_, index = X_train.columns)
print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")
imp_coef = pd.concat([coef.sort_values().head(10),

                     coef.sort_values().tail(10)])
matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)

imp_coef.plot(kind = "barh")

plt.title("Coefficients in the Lasso Model");
import xgboost as xgb
dtrain = xgb.DMatrix(X_train, label = y)

dtest = xgb.DMatrix(X_test)



params = {"max_depth":4, "eta":0.1}

model = xgb.cv(params, dtrain,  num_boost_round=500, early_stopping_rounds=100)
model.loc[30:,["test-rmse-mean", "train-rmse-mean"]].plot()
model_xgb = xgb.XGBRegressor(n_estimators=360, max_depth=2, learning_rate=0.1) #the params were tuned using xgb.cv

model_xgb.fit(X_train, y)
from sklearn.grid_search import GridSearchCV
cv_params = {'max_depth': [3,5,7], 'min_child_weight': [1,3,5]}

ind_params = {'learning_rate': 0.1, 'n_estimators': 1000, 'seed':1301, 'subsample': 0.8, 'colsample_bytree': 0.8, 

             'objective': 'reg:linear'}

optimized_GBM = GridSearchCV(xgb.XGBRegressor(**ind_params), 

                            cv_params, 

                             scoring = 'neg_mean_squared_error', cv = 5, n_jobs = -1) 
optimized_GBM.fit(X_train, y)
optimized_GBM.grid_scores_
model_xgb = xgb.XGBRegressor(n_estimators=360, max_depth=3, learning_rate=0.1, min_child_weight=3) #the params were tuned using xgb.cv

model_xgb.fit(X_train, y)
xgb_preds = np.polyval(p, test['OverallQual']) + model_xgb.predict(X_test)
solution = pd.DataFrame({"id":test.Id, "SalePrice":xgb_preds})

solution.to_csv("poly_xgb.csv", index = False)
solution.head()