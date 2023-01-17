import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import scipy.stats as stats

import warnings

warnings.filterwarnings('ignore')
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.head()
plt.subplots(figsize=(8, 6))

sns.distplot(train['SalePrice'], kde = False, fit = stats.norm)
prob = stats.probplot(train['SalePrice'], plot=plt)
print('Skewness: %f' % train['SalePrice'].skew())

print('Kurtosis: %f' % train['SalePrice'].kurt())
train['SalePrice'] = np.log1p(train['SalePrice'])
plt.subplots(figsize=(8, 4))

sns.distplot(train['SalePrice'], kde = False, fit = stats.norm)

plt.figure()

prob = stats.probplot(train['SalePrice'], plot=plt)
print('Skewness: %f' % train['SalePrice'].skew())

print('Kurtosis: %f' % train['SalePrice'].kurt())
corrmat = train.corr(method='spearman')

plt.subplots(figsize=(12, 9))



k = 15 #number of variables for heatmap

cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(train[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 8}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
sns.set()

cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'YearBuilt', 'GarageArea', 'FullBath', 'TotalBsmtSF']

sns.pairplot(train[cols], size = 2.5)

plt.show();
data = pd.concat([train['SalePrice'], train['OverallQual']], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.boxplot(x='OverallQual', y="SalePrice", data=data)
data = pd.concat([train['SalePrice'], train['GarageCars']], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.boxplot(x='GarageCars', y="SalePrice", data=data)
data = pd.concat([train['SalePrice'], train['FullBath']], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.boxplot(x='FullBath', y="SalePrice", data=data)
all_data = pd.concat((train.iloc[:, 1:-1], test.iloc[:, 1:]))
missing = all_data.isnull().sum()

missing = missing[missing > 0]

missing.sort_values(inplace=True)

missing.plot.bar()

missing_data = pd.DataFrame({'Total': missing})

missing_data.sort_values(by='Total',ascending=False)
all_data.shape
all_data = all_data.drop((missing_data[missing_data['Total'] > 100]).index,1)
from sklearn.preprocessing import OneHotEncoder

all_data = pd.get_dummies(all_data)
all_data = all_data.fillna(all_data.mean())
all_data.shape
all_data.isnull().sum().max()
quan_feats = all_data.dtypes[all_data.dtypes != "object"].index



skewed_feats = all_data[quan_feats].apply(lambda x: stats.skew(x))

skewed_feats = skewed_feats[skewed_feats > 0.5].index



all_data[skewed_feats] = np.log1p(all_data[skewed_feats])
X_train = all_data[:train.shape[0]]

X_test = all_data[train.shape[0]:]

y = train['SalePrice']
from sklearn.linear_model import Ridge, Lasso

from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import RobustScaler



# additionally I produce robust scaling to increase model accuracy

scaler = RobustScaler().fit_transform(X_train) 



#creating the cross validation function for ridge and lasso

def rmse_cv(model):

    rmse= np.sqrt(-cross_val_score(model, scaler, y, scoring="neg_mean_squared_error", cv = 5))

    return(rmse)
alphas_r = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]



val_errors_r = []

for alpha in alphas_r:

    ridge = Ridge(alpha = alpha)

    errors_r = rmse_cv(ridge).mean()

    val_errors_r.append(errors_r)

plt.plot(alphas_r, val_errors_r)

plt.title('Ridge')

plt.xlabel('lambda')

plt.ylabel('rmse')
print('best alpha: {}'.format(alphas_r[np.argmin(val_errors_r)]))

print('Min RMSE: {}'.format(min(val_errors_r)))
alphas_l = [0.00005, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005,

           0.0006, 0.0007, 0.0008, 1e-3, 5e-3]



val_errors_l = []

for alpha in alphas_l:

    lasso = Lasso(alpha = alpha)

    errors_l = rmse_cv(lasso).mean()

    val_errors_l.append(errors_l)
plt.plot(alphas_l, val_errors_l)

plt.title('Lasso')

plt.xlabel('alpha')

plt.ylabel('rmse')
print('best alpha: {}'.format(alphas_l[np.argmin(val_errors_l)]))

print('Min RMSE: {}'.format(min(val_errors_l)))
import xgboost as xgb
#Create a train and test matrix for xgb

dtrain = xgb.DMatrix(data = X_train, label = y)

dtest = xgb.DMatrix(X_test)
untuned_params = {'objective':'reg:linear'}

untuned_cv = xgb.cv(dtrain = dtrain, params = untuned_params, nfold = 4, metrics='rmse', as_pandas=True, seed = 5)
print('Untuned rmse: %f' % (untuned_cv["test-rmse-mean"].tail(1).values[0]))
gbm_param_grid = {

    'colsample_bytree': [0.3],

#    'subsample': [0.3,0.5, 0.7, 1],

    'n_estimators': [400, 450, 500],

    'max_depth': [3],

    'learning_rate' : [0.1]

}
gbm = xgb.XGBRegressor()
from sklearn.model_selection import GridSearchCV

grid_mse = GridSearchCV(param_grid = gbm_param_grid, estimator = gbm, scoring="neg_mean_squared_error", cv = 4)
grid_mse.fit(X_train,y)
print("Best parameters found: ", grid_mse.best_params_)

print("Lowest RMSE found: ", np.sqrt(np.abs(grid_mse.best_score_)))
tuned_params = {'objective':'reg:linear', 'n_estimators': 450, 'learning_rate': 0.1, 'max_depth': 3, 'colsample_bytree': 0.3, 'subsample': 1}

tuned_cv = xgb.cv(dtrain = dtrain, params = tuned_params, nfold = 4, num_boost_round = 500, metrics='rmse', as_pandas=True, seed = 5)
print('Tuned rmse: %f' % (tuned_cv["test-rmse-mean"].tail(1).values[0]))
l2_params = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]

tuned_params = {'objective':'reg:linear', 'n_estimators': 450, 'learning_rate': 0.1, 'max_depth': 3, 'colsample_bytree': 0.3, 'subsample': 1}
rmses_l2 = []
for reg in l2_params:

    tuned_params['lambda'] = reg

    cv_results_rmse = xgb.cv(tuned_params,dtrain, num_boost_round=500, early_stopping_rounds=100, nfold=4, metrics ='rmse', as_pandas=True, seed =123)

    rmses_l2.append(cv_results_rmse['test-rmse-mean'].tail(1).values[0])
print("Best rmse as a function of l2:")

print(pd.DataFrame(list(zip(l2_params, rmses_l2)), columns=["l2", "rmse"]), '\n')

print('Min L2 Tuned rmse: %f' % (min(rmses_l2)))

print('Min lambda: %f' % (min(l2_params)))
l1_params = [0.00005, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 1e-3, 5e-3]

tuned_params = {'objective':'reg:linear', 'n_estimators': 450, 'learning_rate': 0.1, 'max_depth': 3, 'colsample_bytree': 0.3, 'subsample': 1}
rmses_l1 = []
for reg in l1_params:

    tuned_params['alpha'] = reg

    cv_results_rmse = xgb.cv(tuned_params,dtrain, num_boost_round=500, early_stopping_rounds=100, nfold=4, metrics ='rmse', as_pandas=True, seed =123)

    rmses_l1.append(cv_results_rmse['test-rmse-mean'].tail(1).values[0])
print("Best rmse as a function of l2:")

print(pd.DataFrame(list(zip(l1_params, rmses_l1)), columns=["l1", "rmse"]), '\n')

print('Min L1 Tuned rmse: %f' % (min(rmses_l1)))

print('Min alpha: %f' % (min(l1_params)))
model_xgb = xgb.XGBRegressor(objective= 'reg:linear',reg_alpha = 0.00005, n_estimators=500, learning_rate=0.1, max_depth=3, colsample_bytree=0.3, subsample=1) 

model_xgb.fit(X_train, y)
xgb_preds = np.expm1(model_xgb.predict(X_test))
lasso.fit(X_train, y)

lass_pred = np.expm1(lasso.predict(X_test))
preds = xgb_preds
solution = pd.DataFrame({"id":test.Id, "SalePrice":preds})

solution.to_csv("sol5.csv", index = False)