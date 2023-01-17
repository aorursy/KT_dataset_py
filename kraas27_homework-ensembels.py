import numpy as np

import pandas as pd

import matplotlib

import matplotlib.pyplot as plt

%matplotlib inline ()

%config InlineBackend.figure_format = 'svg' 

from pylab import rcParams

rcParams['figure.figsize'] = (9, 6)

import warnings

warnings.filterwarnings("ignore")
'''sample = pd.read_csv('sample_submission.csv') # пример ответа для kaggle

sample.head()'''
data_train = pd.read_csv('../input/train.csv')

data_train.head()
'''data_test = pd.read_csv('test.csv') # для kaggle

data_test.head()'''
# разбиваем данные на train и на test

data_train_y = data_train['SalePrice'].copy()

data_train_x = data_train.drop('SalePrice', axis=1)
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(data_train_x, data_train_y, test_size=0.3)  
X_train.shape, y_train.shape, X_valid.shape, y_valid.shape
# Соединим train и test, для удобства обработки

X_valid['is_test'] = 1

X_train['is_test'] = 0
data = pd.concat([X_valid, X_train])
data.info()
data.isnull().sum()[data.isnull().sum().values != 0]
#Находим категориальные признаки

cat_feat = list(data.dtypes[data.dtypes == object].index)
# Разбиваем данные на категориальные и числовые

data_cat = data[cat_feat]



data_int = data.drop(cat_feat, axis=1)
# заполняем пропуски

data[data_cat.columns] = data_cat[data_cat.columns].fillna('NaN')

data[data_int.columns] = data[data_int.columns].fillna(0)
data.isnull().sum()[data.isnull().sum().values != 0]
# Обрабатываем категориальные переменные (при предположении что в test будут отличные значения,следовало бы обучаться только на train)

# Так же для каждого признака нужно было бы создать LabelEncoder, но в нашем примере можно этого не делать

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import LabelEncoder 

le = LabelEncoder()



for i  in data_cat.columns:

    if len(data[i].unique()) < 8:

        data = pd.get_dummies(data, columns=[i])

    else:

        le.fit(data[i])

        data[i] = le.transform(data[i])
data.info()
data.head(3)
# делим данные на трейн и тест

X_train = data[data.is_test==0]

X_valid = data[data.is_test==1]



X_train = X_train.drop('is_test', axis=1)

X_valid = X_valid.drop('is_test', axis=1)
# делаем нормирование

from sklearn.preprocessing import StandardScaler

std = StandardScaler()

X_train_sc = std.fit_transform(X_train)

X_train = pd.DataFrame(X_train_sc, columns=X_train.columns)



X_valid_sc = std.transform(X_valid)

X_valid = pd.DataFrame(X_valid_sc, columns=X_train.columns)
X_train.shape, y_train.shape, X_valid.shape, y_valid.shape
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import GridSearchCV
clf_rf = RandomForestRegressor(n_jobs=-1, n_estimators=200)

max_depth = [5, 9, 13, 15]

min_samples_split = [10, 15, 25]

param_grid = {'max_depth': max_depth, 'min_samples_split': min_samples_split}



clf_rf = GridSearchCV(clf_rf, param_grid, cv=10, n_jobs=-1, scoring='neg_mean_squared_error')

clf_rf.fit(X_train, y_train)
clf_rf_best = clf_rf.best_estimator_
y_pred_rf = clf_rf_best.predict(X_valid)
# Оценка по RandomForestRegressor

from sklearn.metrics import mean_squared_error

lin_mse = mean_squared_error(y_valid, y_pred_rf)

lin_rmse_rf = np.sqrt(lin_mse)

lin_rmse_rf
# Выводим важность признаков (TOP-5) 

imp = pd.Series(clf_rf_best.feature_importances_, index=X_train.columns)

imp.sort_values(ascending=False).head()
from sklearn.model_selection import StratifiedKFold
cv = StratifiedKFold(n_splits=10)
def get_meta_features(clf, X_train, y_train, X_test, stack_cv):

    meta_train = np.zeros_like(y_train, dtype=float) 

    meta_test = np.zeros(X_test.shape[0], dtype=float) 

    

    for i, (train_ind, test_ind) in enumerate(stack_cv.split(X_train, y_train)): 

        

        clf.fit(X_train.iloc[train_ind], y_train.iloc[train_ind])

        meta_train[test_ind] = clf.predict(X_train.iloc[test_ind])

        meta_test += clf.predict(X_test)

    

    return meta_train, meta_test / stack_cv.n_splits
from sklearn.linear_model import Lasso

lin_reg = Lasso()
param_grid = {'alpha': [0.01, 0.05, 0.1, 0.5, 1.]}

lasso_grid = GridSearchCV(lin_reg, param_grid, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')
lasso_grid.fit(X_train, y_train)
lasso_grit_best = lasso_grid.best_estimator_
y_pred_lasso = lasso_grit_best.predict(X_valid)
lin_mse = mean_squared_error(y_valid, y_pred_lasso)

rmse_lasso = np.sqrt(lin_mse)

rmse_lasso
clf_rf_best = clf_rf.best_estimator_
y_pred_rf = clf_rf_best.predict(X_valid)
lin_mse = mean_squared_error(y_valid, y_pred_rf)

rmse_rf = np.sqrt(lin_mse)

rmse_rf
import xgboost as xgb
xgb_model_2 = xgb.XGBRegressor(n_estimators=30, n_jobs=-1)
# Подбор параметров закоментирован, так как слишком долго вычисляет

'''

max_depth = [2, 5, 7, 9]

colsample_bytree = [0.3, 0.6, 0.9, 1]

subsample = [0.3, 0.6, 0.9, 1]

gamma = [0, 0.3, 0.6, 0.8, 1]

alpha = [0, 0.3, 0.6, 0.9]

lamba = [0.1, 0.3, 0.6, 0.8, 1]



params = {

    'max_depth' : max_depth,

    'colsample_bytree' : colsample_bytree,

    'subsample' : subsample,

    'gamma' : gamma,

    'reg_alpha' : alpha,

    'reg_lambda' : lamba

}



gr_xgb = GridSearchCV(xgb_model_2, params, n_jobs=-1, cv=10, scoring='neg_mean_squared_error')

xgb_model = gr_xgb.fit(X_train, y_train)



xgb_model.best_params_



>>> 'colsample_bytree': 1,

>>> 'gamma': 0,

>>> 'max_depth': 5,

>>> 'reg_alpha': 0.6,

>>> 'reg_lambda': 0.1,

>>> 'subsample': 0.9

'''
'''

xgb_model_3 = xgb.XGBRegressor(colsample_bytree=1, gamma=0, reg_alpha=0.6, reg_lambda=0.1,

                                   subsample=0.9, n_jobs=-1, n_estimators=5000, learning_rate=0.1)





learning_rate = [0.005, 0.007, 0.01]



fit_params={"early_stopping_rounds":50, 

            "eval_metric" : "rmse", 

            "eval_set" : [[X_train, y_train]]}



params = {

    'learning_rate' : learning_rate

}



gr_xgb_ = GridSearchCV(xgb_model_3, params, fit_params=fit_params, n_jobs=-1, cv=10, 

                       scoring='neg_mean_squared_error')



gr_xgb_.fit(X_train, y_train)



xgb_best = gr_xgb_.best_estimator_

print (xgb_best.best_ntree_limit)

print (gr_xgb_.best_params_)



>>> 5000

>>> {'learning_rate': 0.01}

'''
xgb_model_final = xgb.XGBRegressor(colsample_bytree=1, gamma=0, max_depth=5, reg_alpha=0.6, reg_lambda=0.1,

                                   subsample=0.9, n_jobs=-1, n_estimators=5000, learning_rate=0.01)
xgb_model_final.fit(X_train, y_train)
y_pred_xgb = xgb_model_final.predict(X_valid)
rmse_xgb = np.sqrt(mean_squared_error(y_valid, y_pred_xgb))

rmse_xgb
all_models = [lasso_grit_best, clf_rf_best, xgb_model_final]
meta_train = []

meta_test = []

for i in all_models:

    meta_tr, meta_te = get_meta_features(i, X_train, y_train, X_valid, cv)

    meta_train.append(meta_tr)

    meta_test.append(meta_te)



    

col_names = ['lasso_model', 'rf_model', 'xgb_model']

X_train_meta = pd.DataFrame(np.stack(meta_train, axis=1), columns=col_names)

X_test_meta = pd.DataFrame(np.stack(meta_test,axis=1), columns=col_names)
X_train_meta.head(3)
from sklearn.linear_model import LinearRegression

clf_lr_meta = LinearRegression(n_jobs=-1)
clf_lr_meta.fit(X_train_meta, y_train)
yl_pred_meta_test = clf_lr_meta.predict(X_test_meta)
lin_mse = mean_squared_error(y_valid, yl_pred_meta_test)

rmse_stack = np.sqrt(lin_mse)

rmse_stack
pd.Series(clf_lr_meta.coef_.flatten(), index=X_train_meta.columns).plot(kind='barh')
print ('Результат lasso_reg',rmse_lasso)

print ('Результат Lasso RandomForest',rmse_rf)

print ('Результат XGBoost',rmse_xgb)

print ('Результат stacking',rmse_stack)