import gc

import warnings

warnings.filterwarnings('ignore')



import numpy as np

import scipy as sp

import pandas as pd

from pandas import DataFrame, Series



from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

from sklearn.metrics import roc_auc_score, mean_squared_error, mean_squared_log_error, log_loss

from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold

from sklearn.preprocessing import LabelEncoder



from tqdm import tqdm_notebook as tqdm

from category_encoders import OrdinalEncoder



import matplotlib.pyplot as plt

plt.style.use('ggplot')

%matplotlib inline



import lightgbm as lgb

from lightgbm import LGBMRegressor

import pandas_profiling
df_train = pd.read_csv('../input/exam-for-students20200129/train.csv', index_col=0)#, skiprows=lambda x: x%20!=0)

#df_test = #testデータの読み込みをtrainを参考に書いて見ましょう！

X_test = pd.read_csv('../input/exam-for-students20200129/test.csv', index_col=0)#, skiprows=lambda x: x%20!=0)
df_train = df_train[['ConvertedSalary','Employment', 'LastNewJob', 'Country', 'YearsCodingProf', 'Currency', 'SalaryType', 'RaceEthnicity', 'Age', 'DevType', 'CompanySize']]
X_test = X_test[['Employment', 'LastNewJob', 'Country', 'YearsCodingProf', 'Currency', 'SalaryType', 'RaceEthnicity', 'Age', 'DevType', 'CompanySize']]
#pandas_profiling.ProfileReport(X_test)
df_train.head()
y_train = df_train.ConvertedSalary

X_train = df_train.drop(['ConvertedSalary'], axis=1)
X_train = X_train[y_train.isnull()==False]

y_train = y_train[y_train.isnull()==False]
from category_encoders import OrdinalEncoder



cats = []



for col in X_train.columns:

    if X_train[col].dtype == 'object':

        cats.append(col)

        

oe = OrdinalEncoder(cols=cats)



X_train = oe.fit_transform(X_train)

X_test = oe.transform(X_test)
X_train.fillna(-99999, inplace=True)

X_test.fillna(-99999, inplace=True)
X_train.head()
X_test.head()
%%time

# 金額系なので、RMSLEで最適化してみる。

# 何も指定しないと大抵はRMSEで最適化される。ここでは対数を取っておくとちょうどRMSLEの最適化に相当する。

scores = []



skf = KFold(n_splits=5, random_state=71, shuffle=True)



for i, (train_ix, test_ix) in enumerate(tqdm(skf.split(X_train, y_train))):

    X_train_, y_train_ = X_train.values[train_ix], y_train.values[train_ix]

    X_val, y_val = X_train.values[test_ix], y_train.values[test_ix]



    clf = LGBMRegressor(boosting_type='gbdt', 

                             class_weight=None, 

                             colsample_bytree=0.9,

                             importance_type='split', 

                             learning_rate=0.05, 

                             max_depth=-1,

                              min_child_samples=20, 

                             min_child_weight=0.001, 

                             min_split_gain=0.0,

                               n_estimators=9999, 

                             n_jobs=-1, 

                             num_leaves=15,

                             objective='regression',

                               random_state=71, 

                             reg_alpha=0.0, 

                             reg_lambda=0.0, 

                             silent=True,

                                subsample=1.0, subsample_for_bin=200000, subsample_freq=0)

    

    clf.fit(X_train_, np.log1p(y_train_), early_stopping_rounds=20, eval_metric='RMSLE', eval_set=[(X_val, y_val)])



#    y_pred=clf.predict(X_val)

    y_pred = np.expm1(clf.predict(X_val))

    for i, y_pred_a in enumerate(y_pred):

        if (y_pred_a < 0).any():

            y_pred[i] = 0

            

    #対数を取る



       



  #  if (y_pred < 0).any():

   #        y_pred = null

  #  if (y_pred < 0).any(): #check for negative values

  #          continue

            

    score = mean_squared_log_error(y_val, y_pred)**0.5

    scores.append(score)

    

    

    print('CV Score of Fold_%d is %f' % (i, score))
print(np.mean(scores))

print(scores)
# Fold_4のactual vs predをプロットしてみよう

plt.figure(figsize=[7,7])

plt.scatter(y_val, y_pred, s=5)

plt.xlabel('actual')

plt.ylabel('pred')

plt.show()
imp = DataFrame(clf.booster_.feature_importance(importance_type='gain'), index = X_train.columns, columns=['importance']).sort_values(['importance'], ascending=False)

imp
imp.shape
use_col = imp.index[:10]

use_col
fig, ax = plt.subplots(figsize=(5, 8))

lgb.plot_importance(clf, max_num_features=50, ax=ax, importance_type='gain')
# 全データで再学習し、testに対して予測する

clf.fit(X_train, y_train)



#y_pred = clf.predict_proba(X_test)[:,1]

y_pred = clf.predict(X_test)

#y_pred = np.expm1(clf.predict(X_test))
y_pred
# こちらもスムーズな進行のために20分の１に間引いていますが、本番では"skiprows=lambda x: x%20!=0"を削除して用いてください。

submission = pd.read_csv('../input/exam-for-students20200129/sample_submission.csv', index_col=0)#, skiprows=lambda x: x%20!=0)



submission.ConvertedSalary = y_pred

submission.to_csv('submission.csv')
submission.head()