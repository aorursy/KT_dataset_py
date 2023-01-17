import pandas as pd
import numpy as np
import catboost as cb
import lightgbm as lgb
import time

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import average_precision_score
X = pd.read_csv('../input/application_train.csv')
y = X['TARGET']
X = X.drop(['TARGET'], axis=1)
print(X.columns)
X = X.fillna(0)
categorical_features = []
numerical_features = []
for column in X.columns:
    if (X[column].dtype == 'O'):
        categorical_features = categorical_features + [column]
    else:
        numerical_features = numerical_features + [column]
print(len(categorical_features))        
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7, random_state = 0)
st_time = time.time()
model = cb.CatBoostClassifier(task_type = "GPU")
model.fit(X_train[numerical_features], y_train)
print('CatBoost:')
print('time:', time.time() - st_time)
print(average_precision_score(y_test, model.predict_proba(X_test[numerical_features])[:, 1]))
st_time = time.time()
model = lgb.LGBMClassifier()
model.fit(X_train[numerical_features], y_train)
print('Lgb:')
print('time:', time.time() - st_time)
print(average_precision_score(y_test, model.predict_proba(X_test[numerical_features])[:, 1]))
#st_time = time.time()
#cat_boost_params = {
#    'depth': [2,5,8],
#    'n_estimators':[200, 500, 1000],
#    'learning_rate': [0.03, 0.05, 0.1],
#    'loss_function': ['Logloss', 'CrossEntropy']
#}
#mod_cb = GridSearchCV(cb.CatBoostClassifier(task_type="GPU"), cv=3, param_grid=cat_boost_params, scoring='average_precision')
#mod_cb.fit(X_train[numerical_features], y_train)
#print('CbGridSearch')
#print('time:', time.time() - st_time)
#print(mod_cb.best_params_)
st_time = time.time()
lgbm_params = {
    'max_depth': [2,5,8],
    'n_estimators':[200, 500, 1000],
    'learning_rate': [0.03, 0.05, 0.1],
    'metric': ['binary', 'regression']
}
mod_lgbm = GridSearchCV(lgb.LGBMClassifier(), cv=3, param_grid=lgbm_params, scoring='average_precision')
mod_lgbm.fit(X_train[numerical_features], y_train)
print('CbGridSearch')
print('time:', time.time() - st_time)
print(mod_lgbm.best_params_)
### ╰( ͡° ͜ʖ ͡° )つ──☆*:・ﾟ
### ╰( ͡° ͜ʖ ͡° )つ──☆*:・ﾟ
### ╰( ͡° ͜ʖ ͡° )つ──☆*:・ﾟ