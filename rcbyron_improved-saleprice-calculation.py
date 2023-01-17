import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import Imputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import RidgeCV, LassoCV, lasso_path
from scipy.stats import skew
from xgboost import XGBRegressor

# Import data
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

# Examine training data
corr_matrix = train.corr()
sns.heatmap(corr_matrix, vmax=0.9);

# Preprocessing
# train.dropna(axis=1, how='any', inplace=True)
# test.dropna(axis=1, how='any', inplace=True)

train['SalePrice'] = np.log1p(train['SalePrice'])

# Log transform skewed numeric features
# numeric_feats = train.dtypes[train.dtypes != "object"].index
# skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) # compute skewness
# skewed_feats = skewed_feats[skewed_feats > 0.75]
# skewed_feats = skewed_feats.index
# train[skewed_feats] = np.log1p(train[skewed_feats])
# train = pd.get_dummies(train)

# numeric_feats = test.dtypes[test.dtypes != "object"].index
# skewed_feats = test[numeric_feats].apply(lambda x: skew(x.dropna())) # compute skewness
# skewed_feats = skewed_feats[skewed_feats > 0.75]
# skewed_feats = skewed_feats.index
# test[skewed_feats] = np.log1p(test[skewed_feats])
# test = pd.get_dummies(test)

train = train.fillna(train.mean())
test = test.fillna(test.mean())

train_cols = list(test.columns) + ['SalePrice']
train = train.filter(train_cols, axis=1)
test = test.filter(train.columns, axis=1)

Y = train.SalePrice
X = train.drop(['SalePrice'], axis=1).select_dtypes(exclude=['object'])
full_trainY = Y.as_matrix()
full_trainX = X.as_matrix()
full_testX = test.select_dtypes(exclude=['object']).as_matrix()
trainX, testX, trainY, testY = train_test_split(X.as_matrix(), Y.as_matrix(), test_size=0.25)

# Filling NA's with the mean of the column
full_trainX = imp.fit_transform(full_trainX)
full_testX = imp.transform(full_testX)

def evaluate_accuracy(testY, predictions):
    """Make predictions and print errors."""
    print("MAE: ", mean_absolute_error(testY, predictions))
    print("RMSE:", mean_squared_error(np.log(testY), np.log(predictions))**0.5)
    print("R2:", r2_score(testY, predictions))
# Ridge Regression
model_ridge = RidgeCV(alphas=(0.1, 1.0, 10.0), fit_intercept=True, normalize=True,
                      scoring=None, cv=None, gcv_mode=None, store_cv_values=False)
model_ridge.fit(trainX, trainY)
p = model_ridge.predict(testX)
evaluate_accuracy(testY, p)
# Lasso Regression
model_lasso = LassoCV(eps=0.0001, n_alphas=1000, alphas=None, fit_intercept=True,
                      normalize=True, precompute='auto', max_iter=10000, tol=0.0001,
                      copy_X=True, cv=None, verbose=False, n_jobs=1, positive=True,
                      random_state=None, selection='cyclic')
model_lasso.fit(trainX, trainY)
p = model_lasso.predict(testX)
print(np.sum(np.isnan(p)))
evaluate_accuracy(testY, p)
# Plot alphas vs. l0 norms of each vector
alphas, coefs, _ = LassoCV.path(trainX, trainY)
l0_norms = coefs.shape[0] - np.count_nonzero(coefs, axis=0)
plt.scatter(alphas, l0_norms)
# XGBoost
model_xgb = XGBRegressor(max_depth=3, learning_rate=0.1, n_estimators=1000, silent=True,
                         objective='reg:linear', booster='gbtree', n_jobs=1, nthread=None,
                         gamma=0, min_child_weight=1, max_delta_step=0, subsample=1,
                         colsample_bytree=1, colsample_bylevel=1, reg_alpha=0, reg_lambda=1,
                         scale_pos_weight=1, base_score=0.5, random_state=0, seed=None,
                         missing=None)
model_xgb.fit(trainX, trainY, early_stopping_rounds=5, 
              eval_set=[(testX, testY)], verbose=False)
p = model_xgb.predict(testX)
evaluate_accuracy(testY, p)

# Submit file
new_p = model_xgb.predict(full_testX)
my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': new_p})
my_submission.to_csv('submission.csv', index=False)
print(my_submission['SalePrice'].describe())
# Base layer trainX concatenation
train_saleprice_ridge = model_ridge.predict(trainX)
train_saleprice_lasso = model_lasso.predict(trainX)
new_trainX = np.column_stack((train_saleprice_ridge, train_saleprice_lasso))

# Base layer testX concatenation
test_saleprice_ridge = model_ridge.predict(testX)
test_saleprice_lasso = model_lasso.predict(testX)
new_testX = np.column_stack((test_saleprice_ridge, test_saleprice_lasso))

# Second layer
stacked_ridge = RidgeCV(alphas=(0.1, 1.0, 10.0), fit_intercept=True, normalize=False,
                        scoring=None, cv=None, gcv_mode=None, store_cv_values=False)
stacked_ridge.fit(new_trainX, trainY)
p = stacked_ridge.predict(new_testX)
evaluate_accuracy(testY, p)

# Base layer trainX concatenation
train_saleprice_ridge = model_ridge.predict(trainX)
train_saleprice_lasso = model_lasso.predict(trainX)
train_saleprice_xgb = model_xgb.predict(trainX)
new_trainX = np.column_stack((train_saleprice_ridge, train_saleprice_lasso, train_saleprice_xgb))

# Base layer testX concatenation
test_saleprice_ridge = model_ridge.predict(testX)
test_saleprice_lasso = model_lasso.predict(testX)
test_saleprice_xgb = model_xgb.predict(testX)
new_testX = np.column_stack((test_saleprice_ridge, test_saleprice_lasso, test_saleprice_xgb))

# Second layer
stacked_ridge = RidgeCV(alphas=(0.1, 1.0, 10.0), fit_intercept=True, normalize=False,
                        scoring=None, cv=None, gcv_mode=None, store_cv_values=False)
stacked_ridge.fit(new_trainX, trainY)
p = stacked_ridge.predict(new_testX)
evaluate_accuracy(testY, p)
# Base layer full_trainX concatenation
model_ridge = RidgeCV(alphas=(0.1, 1.0, 10.0), fit_intercept=True, normalize=True,
                      scoring=None, cv=None, gcv_mode=None, store_cv_values=False)
model_ridge.fit(full_trainX, full_trainY)

model_lasso = LassoCV(eps=0.0001, n_alphas=1000, alphas=None, fit_intercept=True,
                      normalize=False, precompute='auto', max_iter=10000, tol=0.0001,
                      copy_X=True, cv=None, verbose=False, n_jobs=1, positive=True,
                      random_state=None, selection='cyclic')
model_lasso.fit(full_trainX, full_trainY)

full_train_saleprice_ridge = model_ridge.predict(full_trainX)
full_train_saleprice_lasso = model_lasso.predict(full_trainX)
full_train_saleprice_xgb = model_xgb.predict(full_trainX)
new_full_trainX = np.column_stack((full_train_saleprice_ridge, full_train_saleprice_lasso, full_train_saleprice_xgb))

# Base layer full_testX
full_test_saleprice_ridge = model_ridge.predict(full_testX)
full_test_saleprice_lasso = model_lasso.predict(full_testX)
full_test_saleprice_xgb = model_xgb.predict(full_testX)
new_full_testX = np.column_stack((full_test_saleprice_ridge, full_test_saleprice_lasso, full_test_saleprice_xgb))

# Second layer
stacked_ridge = RidgeCV(alphas=(0.1, 1.0, 10.0), fit_intercept=True, normalize=True,
                        scoring=None, cv=None, gcv_mode=None, store_cv_values=False)
stacked_ridge.fit(new_full_trainX, full_trainY)

# Submit file
new_p = stacked_ridge.predict(new_full_testX)
new_p = np.expm1(new_p)
my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': new_p})
my_submission.to_csv('submission.csv', index=False)
print(new_p)
print(my_submission['SalePrice'].describe())