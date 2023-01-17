import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
df_train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv', index_col='Id')
df_test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv', index_col='Id')
df_train.shape, df_test.shape
pd.options.display.max_columns = 80
df_train.head(10)
df_train.describe()
X_train, y_train = df_train.drop('SalePrice', axis=1), df_train['SalePrice']
X_test = df_test
numbers = ['int64', 'float64']
num_features = X_train.select_dtypes(include=numbers).columns
cat_features = X_train.select_dtypes(exclude=numbers).columns

print (f'num features:\n%s'% (num_features))

print (f'cat features:\n%s'% (cat_features))
# this nans should be removed

X_train[num_features].isna().sum()
# here all nans mean 0 (no such feature)

X_train[cat_features].isna().sum()
plt.subplots(figsize=(30, 20))
sns.heatmap(X_train.corr(), vmax=0.9, square=True, annot=True);
nums_train = X_train[num_features].fillna(0)
cats_train = X_train[cat_features].fillna('None')
X_train = pd.concat([nums_train, cats_train], axis=1)

nums_test = X_test[num_features].fillna(0)
cats_test = X_test[cat_features].fillna('None')
X_test = pd.concat([nums_test, cats_test], axis=1)

X_train
ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')

encoded_features = pd.DataFrame(ohe.fit_transform(X_train[cat_features]))

dummy_cols_names = []
for ohe_cat, cat in zip(ohe.categories_, cat_features):
    dummy_cols_names += [f'{cat}_{categ}' for categ in ohe_cat]

encoded_features.columns = dummy_cols_names

X_train.reset_index(drop=True, inplace=True)
X_train = pd.concat([X_train.drop(cat_features, axis=1), encoded_features], axis=1)

X_train
encoded_features = pd.DataFrame(ohe.transform(X_test[cat_features]))

dummy_cols_names = []
for ohe_cat, cat in zip(ohe.categories_, cat_features):
    dummy_cols_names += [f'{cat}_{categ}' for categ in ohe_cat]

encoded_features.columns = dummy_cols_names

X_test.reset_index(drop=True, inplace=True)
X_test = pd.concat([X_test.drop(cat_features, axis=1), encoded_features], axis=1)
sns.distplot(y_train)
plt.grid(True);
y_train_unsk = np.log1p(y_train)

sns.distplot(y_train_unsk)
plt.grid(True);
lgb = LGBMRegressor()
params_lgb = {'random_state': [42], 'max_depth': [3, 4, 5, 10], 'n_estimators': [100, 200, 300, 350], 
              'reg_alpha': [0, 1e0, 1e-1, 1e1, 1e2], 'num_leaves': [3, 5, 10, 20, 30, 50]}

xgb = XGBRegressor()
params_xgb = {'random_state': [42], 'max_depth': [3, 5, 8, 9, 10], 'n_estimators': [5, 10, 20, 100, 500]}

rf = RandomForestRegressor()
params_rf = {'random_state': [42], 'max_depth': [3, 5, 6, 7, 8], 'n_estimators': [30, 50, 100, 500]}
clf = lgb
params = params_lgb

cv = GridSearchCV(clf, params, scoring='neg_mean_squared_error', cv=2, n_jobs=4)
cv.fit(X_train, y_train_unsk)
print (f"Best rmse:     %f"%(np.sqrt(np.abs(cv.best_score_))))
print ("Best estimator: ", cv.best_estimator_)
clf = cv.best_estimator_
clf.fit(X_train, y_train_unsk)
preds_unsk = clf.predict(X_test)
# skew the predictions
preds = np.exp(preds_unsk) - 1
pd.DataFrame(preds, columns=['SalePrice'], index=df_test.index).to_csv('preds.csv')
sns.distplot(preds_unsk)
plt.grid(True);
sns.distplot(preds)
plt.grid(True);