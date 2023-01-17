# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('../input/santander-customer-satisfaction/train.csv',
                encoding = 'latin-1')
print('dataset shape: ', df.shape)
df.head(3)
df.info()
# label 값인 target 속성 값 분포 알아보기
print(df['TARGET'].value_counts())
unsatisfied_cnt = df[df['TARGET'] == 1].TARGET.count()
total_cnt = df.TARGET.count()
print('unsatisfied 비율은 {0:.2f}'.format((unsatisfied_cnt / total_cnt)))
df.describe()
# 분포를 살펴볼 때 `var3` 변수의 min 값에서 이상치 발견
# 또한 `ID` feature는 단순 식별자이기 때문에 삭제

df['var3'].replace(-999999, df['var3'].mode()[0], inplace=True) # 최빈값으로 대체
df.drop('ID', axis=1, inplace=True)

# feature와 label set 분리
X_features = df.iloc[:, :-1]
y_label = df.iloc[:, -1]
print('feature data shape: {0}'.format(X_features.shape))
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_features,
                                                   y_label,
                                                   test_size = .2, 
                                                   random_state = 0)

train_cnt = y_train.count()
test_cnt = y_test.count()
print('train set shape: {0}, test set shape: {1}'.format(X_train.shape, X_test.shape))

print('train set label 값 분포 비율')
print(y_train.value_counts() / train_cnt)
print('test set label 값 분포 비율')
print(y_test.value_counts() / test_cnt)
# xgboost 학습 모델을 생성하고 예측 결과를 ROC AUC로 평가
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

xgb = XGBClassifier(n_estimators = 500,
                   random_state = 156)

xgb.fit(X_train, y_train,
       early_stopping_rounds = 100,
       eval_metric = 'auc', 
       eval_set = [(X_train, y_train), (X_test, y_test)])

xgb_roc_score = roc_auc_score(y_test, xgb.predict_proba(X_test)[:, 1],
                             average = 'macro')
print('ROC AUC: {0: 4f}'.format(xgb_roc_score))
# XGboost hyper-parameter 튜닝 수행
from sklearn.model_selection import GridSearchCV

# hyper-parameter test의 수행 속도를 향상시키기 위해 n_estimator를 100으로 감소
xgb = XGBClassifier(n_estimator = 100)

params = {
    'max_depth' : [5, 7],
    'min_child_weight' : [1, 3],
    'colsample_bytree' : [0.5, 0.75]
}

gridcv = GridSearchCV(xgb, param_grid = params, cv = 3)
gridcv.fit(X_train, y_train,
          early_stopping_rounds = 30,
          eval_metric = 'auc', 
          eval_set = [(X_train, y_train), (X_test, y_test)])
print('GridSearchCV 최적 파라미터: ', gridcv.best_params_)

xgb_roc_score = roc_auc_score(y_test,
                             gridcv.predict_proba(X_test)[:, 1],
                             average = 'macro')
print('ROC AUC: {0:.4f}'.format(xgb_roc_score))
# 이전에 수정한 파라미터에 더해 다른 파라미터 수정을 진행
xgb = XGBClassifier(n_estimator = 1000,
                   random_state = 156, 
                   learning_rate = .02,
                   max_depth = 7,
                   min_child_weight = 3,
                   colsample_bytree = .75,
                   reg_alpha = .03)

xgb.fit(X_train, y_train,
       early_stopping_rounds = 200,
       eval_metric = 'auc', 
       eval_set = [(X_train, y_train), (X_test, y_test)])

xgb_roc_score = roc_auc_score(y_test, xgb.predict_proba(X_test)[:, 1],
                             average = 'macro')

print('ROC AUC: {0:.4f}'.format(xgb_roc_score))
# feature importance 확인하기
from xgboost import plot_importance

fig, ax = plt.subplots(1, 1, figsize=(10, 8))
plot_importance(xgb, ax=ax,
               max_num_features = 20, 
               height = 0.4)
from lightgbm import LGBMClassifier

lgbm = LGBMClassifier(n_estimator = 500)
evals = [(X_test, y_test)]

lgbm.fit(X_train, y_train,
        early_stopping_rounds = 100,
        eval_metric = 'auc',
        eval_set = evals,
        verbose = True)
lgbm_roc_score = roc_auc_score(y_test, lgbm.predict_proba(X_test)[:, 1],
                               average = 'macro')
print('ROC AUC: {0:.4f}'.format(lgbm_roc_score))
# 이전에 xgboost보다 값이 감소 
# hyper-parameter 조정 진행

lgbm = LGBMClassifier(n_estimators = 200)

params = {
    'num_leaves' : [32 ,64],
    'max_depth' : [128, 160],
    'min_child_samples' : [60, 100],
    'subsample' : [0.8, 1]
}

gridcv = GridSearchCV(lgbm, param_grid = params, cv = 3)
gridcv.fit(X_train, y_train,
          early_stopping_rounds = 30,
          eval_metric = 'auc',
          eval_set = [(X_train, y_train), (X_test, y_test)])
print('GridSearchCV 최적 파라미터: ', gridcv.best_params_)
lgbm_roc_score = roc_auc_score(y_test, gridcv.predict_proba(X_test)[:, 1],
                              average = 'macro')
print('ROC AUC: {0:.4f}'.format(lgbm_roc_score))
lgbm = LGBMClassifier(n_estimator = 1000,
                     num_leaves = 32,
                     sumbsample = 0.8,
                     min_child_samples = 100,
                     max_depth = 128)
evals = [(X_test, y_test)]
lgbm.fit(X_train, y_train,
        early_stopping_rounds = 100,
        eval_metric = 'auc',
        eval_set = evals,
        verbose = True)

lgbm_roc_score = roc_auc_score(y_test, lgbm.predict_proba(X_test)[:, 1],
                               average = 'macro')
print('ROC AUC: {0:.4f}'.format(lgbm_roc_score))
# 제출하기
test = pd.read_csv('../input/santander-customer-satisfaction/test.csv', encoding = 'latin-1')
test_id = test['ID']
test.drop('ID', axis=1, inplace = True)
test.head(3)
pred = lgbm.predict_proba(test)[:, 1]

submission = pd.DataFrame({'ID' : test_id, 'TARGET' : pred})
submission.to_csv('submission.csv', index=False)

print('completed!')