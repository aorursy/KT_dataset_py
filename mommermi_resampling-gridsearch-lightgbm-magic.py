import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.info(max_cols=250)
test.info(max_cols=250)
f, ax = plt.subplots(figsize=(25,5))



train.drop(['target', 'ID_code'], axis=1).plot.box(ax=ax, rot=90)
len(train.loc[train.target == 1])/len(train)
r2 = pd.concat([train.drop(['target', 'ID_code'], axis=1), test.drop('ID_code', axis=1)]).corr()**2

r2 = np.tril(r2, k=-1)  # remove upper triangle and diagonal

r2[r2 == 0] = np.nan # replace 0 with nan
f, ax = plt.subplots(figsize=(20,20))

sns.heatmap(np.sqrt(r2), annot=False,cmap='viridis', ax=ax)
target_r2 = train.drop(['ID_code', 'target'], axis=1).corrwith(train.target).agg('square')



f, ax = plt.subplots(figsize=(25,5))

target_r2.agg('sqrt').plot.bar(ax=ax)
top = target_r2.loc[np.sqrt(target_r2) > 0.048].index

top
from sklearn.preprocessing import PolynomialFeatures



polyfeat_train = pd.DataFrame(PolynomialFeatures(2).fit_transform(train[top]))

polyfeat_test = pd.DataFrame(PolynomialFeatures(2).fit_transform(test[top]))
from imblearn.over_sampling import RandomOverSampler
# additional imports

from imblearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import RobustScaler
from imblearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV

import lightgbm as lgb



lgbpipe = Pipeline([('resample', RandomOverSampler(random_state=42)), ('model', lgb.LGBMClassifier(random_state=42, objective='binary', metric='auc', 

                                                                                                   boosting='gbdt', verbosity=1,

                                                                                                   tree_learner='serial'))])



params = {    

    "model__max_depth" : [20],

    "model__num_leaves" : [30],

    "model__learning_rate" : [0.1],

    "model__subsample_freq": [5],

    "model__subsample" : [0.3],

    "model__colsample_bytree" : [0.05],

    "model__min_child_samples": [100],

    "model__min_child_weight": [10],

    "model__reg_alpha" : [0.12],

    "model__reg_lambda" : [15.5],

    "model__n_estimators" : [600]

    }



# previous best-fit gridsearch parameters and results

# {'model__colsample_bytree': 0.05, 'model__learning_rate': 0.1, 'model__max_depth': 20, 'model__min_child_samples': 100, 'model__min_child_weight': 10, 'model__n_estimators': 100, 'model__num_leaves': 30, 'model__reg_alpha': 0.1, 'model__reg_lambda': 10, 'model__subsample': 0.3, 'model__subsample_freq': 5}

# 0.8735588789424164

# {'model__colsample_bytree': 0.05, 'model__learning_rate': 0.1, 'model__max_depth': 20, 'model__min_child_samples': 100, 'model__min_child_weight': 10, 'model__n_estimators': 400, 'model__num_leaves': 30, 'model__reg_alpha': 0.12, 'model__reg_lambda': 0.2, 'model__subsample': 0.3, 'model__subsample_freq': 5}

# 0.8915905852982839

# {'model__colsample_bytree': 0.05, 'model__learning_rate': 0.1, 'model__max_depth': 20, 'model__min_child_samples': 100, 'model__min_child_weight': 10, 'model__n_estimators': 500, 'model__num_leaves': 30, 'model__reg_alpha': 0.12, 'model__reg_lambda': 0.2, 'model__subsample': 0.3, 'model__subsample_freq': 5}

# 0.8923071245054173

# {'model__colsample_bytree': 0.05, 'model__learning_rate': 0.1, 'model__max_depth': 20, 'model__min_child_samples': 100, 'model__min_child_weight': 10, 'model__n_estimators': 600, 'model__num_leaves': 30, 'model__reg_alpha': 0.12, 'model__reg_lambda': 0.2, 'model__subsample': 0.3, 'model__subsample_freq': 5}

# 0.8925518240005254

# {'model__colsample_bytree': 0.05, 'model__learning_rate': 0.1, 'model__max_depth': 20, 'model__min_child_samples': 100, 'model__min_child_weight': 10, 'model__n_estimators': 550, 'model__num_leaves': 30, 'model__reg_alpha': 0.12, 'model__reg_lambda': 0.2, 'model__subsample': 0.3, 'model__subsample_freq': 5}

# 0.8924978701504809

# {'model__colsample_bytree': 0.05, 'model__learning_rate': 0.1, 'model__max_depth': 20, 'model__min_child_samples': 100, 'model__min_child_weight': 10, 'model__n_estimators': 600, 'model__num_leaves': 30, 'model__reg_alpha': 0.12, 'model__reg_lambda': 15, 'model__subsample': 0.3, 'model__subsample_freq': 5}

# 0.8941148812638564

# {'model__colsample_bytree': 0.05, 'model__learning_rate': 0.1, 'model__max_depth': 20, 'model__min_child_samples': 100, 'model__min_child_weight': 10, 'model__n_estimators': 600, 'model__num_leaves': 30, 'model__reg_alpha': 0.5, 'model__reg_lambda': 12, 'model__subsample': 0.3, 'model__subsample_freq': 5}

# 0.8938169988416745

# {'model__colsample_bytree': 0.05, 'model__learning_rate': 0.1, 'model__max_depth': 20, 'model__min_child_samples': 100, 'model__min_child_weight': 10, 'model__n_estimators': 600, 'model__num_leaves': 30, 'model__reg_alpha': 0.3, 'model__reg_lambda': 15, 'model__subsample': 0.3, 'model__subsample_freq': 5}

# 0.8941407236592286

# {'model__colsample_bytree': 0.05, 'model__learning_rate': 0.1, 'model__max_depth': 20, 'model__min_child_samples': 100, 'model__min_child_weight': 10, 'model__n_estimators': 600, 'model__num_leaves': 30, 'model__reg_alpha': 0.2, 'model__reg_lambda': 15, 'model__subsample': 0.3, 'model__subsample_freq': 5}

# 0.8938875270813017

# {'model__colsample_bytree': 0.05, 'model__learning_rate': 0.1, 'model__max_depth': 20, 'model__min_child_samples': 100, 'model__min_child_weight': 10, 'model__n_estimators': 600, 'model__num_leaves': 30, 'model__reg_alpha': 0.2, 'model__reg_lambda': 15, 'model__subsample': 0.3, 'model__subsample_freq': 5}

# 0.8938875270813017

# {'model__colsample_bytree': 0.05, 'model__learning_rate': 0.1, 'model__max_depth': 20, 'model__min_child_samples': 100, 'model__min_child_weight': 10, 'model__n_estimators': 600, 'model__num_leaves': 30, 'model__reg_alpha': 0.12, 'model__reg_lambda': 15.5, 'model__subsample': 0.3, 'model__subsample_freq': 5}

# 0.8943001048082946

# {'model__colsample_bytree': 0.05, 'model__learning_rate': 0.1, 'model__max_depth': 20, 'model__min_child_samples': 100, 'model__min_child_weight': 10, 'model__n_estimators': 600, 'model__num_leaves': 30, 'model__reg_alpha': 0.12, 'model__reg_lambda': 15.2, 'model__subsample': 0.3, 'model__subsample_freq': 5}

# 0.8939732044413886



lgbgrid = GridSearchCV(lgbpipe, param_grid=params, cv=10, scoring='roc_auc')

lgbgrid.fit(train.drop(['ID_code', 'target'], axis=1), train.target)



print(lgbgrid.best_params_)

print(lgbgrid.best_score_)
from sklearn.linear_model import RidgeClassifier



ridgepipe = Pipeline([('resample', RandomOverSampler(random_state=42)), ('scaler', RobustScaler()), ('model', RidgeClassifier(random_state=42))])



params = {'model__alpha': [1.0]} # between 0.5 and 2; best-fit so far: 1

 

ridgegrid = GridSearchCV(ridgepipe, param_grid=params, cv=3, scoring='roc_auc')

ridgegrid.fit(pd.concat([train.drop(['ID_code', 'target'], axis=1), polyfeat_train], axis=1, join='inner'), train.target)



print(ridgegrid.best_params_)

print(ridgegrid.best_score_)
pred = pd.DataFrame(lgbgrid.predict_proba(test.drop(['ID_code'], axis=1))[:, -1], columns=['target'], index=test.loc[:, 'ID_code'])

pred.to_csv('submission.csv', index=True)
!head submission.csv
test.head()