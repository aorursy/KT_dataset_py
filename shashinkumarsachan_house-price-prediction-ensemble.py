import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib



import matplotlib.pyplot as plt

from scipy.stats import skew

from scipy.stats.stats import pearsonr



from sklearn import metrics

from sklearn.metrics import mean_squared_error
train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],test.loc[:,'MSSubClass':'SaleCondition']))
#log transform the target:

train["SalePrice"] = np.log1p(train["SalePrice"])



#log transform skewed numeric features:

numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness

skewed_feats = skewed_feats[skewed_feats > 0.75]

skewed_feats = skewed_feats.index

skewed_feats

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
#Trying Lasso

model_ridge = Ridge()
alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]

cv_ridge = [rmse_cv(Ridge(alpha = alpha)).mean() 

            for alpha in alphas]
cv_ridge = pd.Series(cv_ridge, index = alphas)

cv_ridge.plot(title = "Validation - Just Do It")

plt.xlabel("alpha")

plt.ylabel("rmse")
cv_ridge.min()
model_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005]).fit(X_train, y)

rmse_cv(model_lasso).mean()
coef = pd.Series(model_lasso.coef_, index = X_train.columns)

print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")

coef
imp_coef = pd.concat([coef.sort_values().head(10),

                     coef.sort_values().tail(10)])
matplotlib.rcParams['figure.figsize'] = (6.0, 3.0)

imp_coef.plot(kind = "barh")

plt.title("Coefficients in the Lasso Model")
#let's look at the residuals as well:

matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)



preds = pd.DataFrame({"preds":model_lasso.predict(X_train), "true":y})

preds["residuals"] = preds["true"] - preds["preds"]

preds.plot(x = "preds", y = "residuals",kind = "scatter")
#Trying XGB

import xgboost as xgb
dtrain = xgb.DMatrix(X_train, label = y)

dtest = xgb.DMatrix(X_test)



params = {"max_depth":2, "eta":0.1}

model = xgb.cv(params, dtrain,  num_boost_round=500, early_stopping_rounds=100)
model.loc[30:,["test-rmse-mean", "train-rmse-mean"]].plot()
model_xgb = xgb.XGBRegressor(n_estimators=360, max_depth=2, learning_rate=0.1)

model_xgb.fit(X_train, y)
xgb_preds = np.expm1(model_xgb.predict(X_test))

lasso_preds = np.expm1(model_lasso.predict(X_test))
predictions = pd.DataFrame({"xgb":xgb_preds, "lasso":lasso_preds})

predictions.plot(x = "xgb", y = "lasso", kind = "scatter")
# trying catboost

from catboost import Pool

from catboost import CatBoostRegressor

from sklearn.model_selection import train_test_split

import numpy as np
X = X_train

test_df = X_test

X.shape, test_df.shape
float_cols = X.dtypes[X.dtypes == "float"].index

X.loc[:,float_cols] = X[float_cols].astype(str)



X_train.shape, X.shape
SEED = 1

X_train.shape,X.shape,y.shape
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, random_state=SEED)



cat_features = list(range(X.shape[1]))
train_data = Pool(data=X_train,

                  label=y_train,

                  cat_features=cat_features

                 )



valid_data = Pool(data=X_valid,

                  label=y_valid,

                  cat_features=cat_features

                 )
%%time



params = { 'early_stopping_rounds': 100,

          'verbose': False,

          'random_seed': SEED,

          'eval_metric':'RMSE'

         }



cbc_7 = CatBoostRegressor(**params)

cbc_7.fit(train_data, 

          eval_set=valid_data, 

          use_best_model=True, 

          plot=True

         );
cbc_7.get_feature_importance(prettified=True)
X_test = test_df

float_cols_test = X_test.dtypes[X_test.dtypes == "float"].index

X_test.loc[:,float_cols_test] = X_test[float_cols_test].astype(str)



X_train.shape, X.shape , X_test.shape
from sklearn.utils.multiclass import type_of_target

type_of_target(y)
%%time



from sklearn.model_selection import StratifiedKFold



n_fold = 3 # amount of data folds

folds = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=SEED)



params = {

          'early_stopping_rounds': 100,

          'verbose': False,

          'random_seed': SEED,

          'eval_metric':'RMSE'

         }



test_data = Pool(data=X_test,

                 cat_features=cat_features)



bins = np.linspace(0, 1, 100) 

y_binned = np.digitize(y, bins)



scores = []

catboost_prediction = np.zeros(X_test.shape[0])

for fold_n, (train_index, valid_index) in enumerate(folds.split(X, y_binned)):

    

    X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]

    y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

    

    train_data = Pool(data=X_train, 

                      label=y_train,

                      cat_features=cat_features)

    valid_data = Pool(data=X_valid, 

                      label=y_valid,

                      cat_features=cat_features)

    

    model = CatBoostRegressor(**params)

    model.fit(train_data,

              eval_set=valid_data, 

              use_best_model=True

             )

    

    score = model.get_best_score()['validation']['RMSE']

    scores.append(score)



    y_pred = model.predict(test_data)

    catboost_prediction += y_pred



catboost_prediction /= n_fold

print('CV mean: {:.4f}, CV std: {:.4f}'.format(np.mean(scores), np.std(scores)))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_valid, model.predict(X_valid))))
cb_preds = np.expm1(catboost_prediction)



predictions = pd.DataFrame({"lasso":lasso_preds, "cb":cb_preds})

predictions.plot(x = "lasso", y = "cb", kind = "scatter")



predictions = pd.DataFrame({"xgb":xgb_preds, "cb":cb_preds})

predictions.plot(x = "xgb", y = "cb", kind = "scatter")



predictions = pd.DataFrame({"xgb":xgb_preds, "lasso":lasso_preds})

predictions.plot(x = "xgb", y = "lasso", kind = "scatter")
#I was not able to run and commit below H20 model and I have ran it seperately and used its output here as input dataset. PLease refer to following link for notebook : couldn'thttps://www.kaggle.com/shashinkumarsachan/house-price-prediction-using-h20-automl



h20_preds = pd.read_csv('../input/house-price-predictions-h20-automl-without-tuning/house_sales_price_pred_full.csv')



predictions = pd.DataFrame({"h20":h20_preds['SalePrice'], "lasso":lasso_preds})

predictions.plot(x = "h20", y = "lasso", kind = "scatter")



predictions = pd.DataFrame({"h20":h20_preds['SalePrice'], "cb":cb_preds})

predictions.plot(x = "h20", y = "cb", kind = "scatter")



predictions = pd.DataFrame({"h20":h20_preds['SalePrice'], "xgb":xgb_preds})

predictions.plot(x = "h20", y = "xgb", kind = "scatter")
# Finally mixing predictions,



preds = 0.57*lasso_preds + 0.0*xgb_preds + 0.0*cb_preds + 0.43*h20_preds['SalePrice']

cb_solution = pd.DataFrame({"id":test.Id, "SalePrice":preds})

cb_solution.to_csv("house_price_preds.csv", index = False)