import numpy as np

import pandas as pd



import warnings

warnings.simplefilter('ignore')

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/train.csv', index_col='id')

test = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/test.csv', index_col='id')
train.head(3).T
def summary(df):

    summary = pd.DataFrame(df.dtypes, columns=['dtypes'])

    summary = summary.reset_index()

    summary['Name'] = summary['index']

    summary = summary[['Name', 'dtypes']]

    summary['Missing'] = df.isnull().sum().values    

    summary['Uniques'] = df.nunique().values

    summary['First Value'] = df.loc[0].values

    summary['Second Value'] = df.loc[1].values

    summary['Third Value'] = df.loc[2].values

    return summary



summary(train)
# 看这一行中有多少个NULL值

train['missing_count'] = train.isnull().sum(axis=1)

test['missing_count'] = test.isnull().sum(axis=1)
missing_number = -99999

missing_string = 'MISSING_STRING'
numerical_features = [

    'bin_0', 'bin_1', 'bin_2',

    'ord_0',

    'day', 'month'

]



string_features = [

    'bin_3', 'bin_4',

    'ord_1', 'ord_2', 'ord_3', 'ord_4', 'ord_5',

    'nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4', 'nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9'

]
def impute(train, test, columns, value):

    for column in columns:

        train[column] = train[column].fillna(value)

        test[column] = test[column].fillna(value)
impute(train, test, numerical_features, missing_number)

impute(train, test, string_features, missing_string)
#ord_5 是由两个字母组成的，这段的目的是将这两个字母拆分成两列，仍然保留用 MISSING_STRING 去填充

train['ord_5_1'] = train['ord_5'].str[0]

train['ord_5_2'] = train['ord_5'].str[1]



train.loc[train['ord_5'] == missing_string, 'ord_5_1'] = missing_string

train.loc[train['ord_5'] == missing_string, 'ord_5_2'] = missing_string



train = train.drop('ord_5', axis=1)





test['ord_5_1'] = test['ord_5'].str[0]

test['ord_5_2'] = test['ord_5'].str[1]



test.loc[test['ord_5'] == missing_string, 'ord_5_1'] = missing_string

test.loc[test['ord_5'] == missing_string, 'ord_5_2'] = missing_string



test = test.drop('ord_5', axis=1)
# 这里将 Feature 作为 3类，simple_features， ohe_features 和 target_features 

# Apply Target to features that have many unique values 这里仅是讲 这一列 Unique Value如果很大的话，才做Target Encoding

simple_features = [

    'missing_count'

]



oe_features = [

    'bin_0', 'bin_1', 'bin_2', 'bin_3', 'bin_4',

    'nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4',

    'ord_0', 'ord_1', 'ord_2', 'ord_3', 'ord_4', 'ord_5_1', 'ord_5_2'

]



target_features = [

    'nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9'

]



cyc_features = ['day', 'month']



# ohe的含义是 One Hot Encoding

ohe_features = [

    'bin_0', 'bin_1', 'bin_2', 'bin_3', 'bin_4',

    'nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4',

    'nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9', # 这一行的这些Feature包含了大量的unique values

    'ord_0', 'ord_1', 'ord_2', 'ord_3', 'ord_4', 'ord_5_1', 'ord_5_2',

    'day', 'month'

]

# 用于 Logit 的是 ohe_features + simple_features

# 用于 Xgboost 的是 oe_features + cyc_features + simple_features + target_features
y_train = train['target'].copy()

x_train = train.drop('target', axis=1)

del train



x_test = test.copy()



del test
from sklearn.preprocessing import StandardScaler





scaler = StandardScaler()

simple_x_train = scaler.fit_transform(x_train[simple_features])

simple_x_test = scaler.transform(x_test[simple_features])
type(simple_x_train)
from sklearn.preprocessing import OneHotEncoder





ohe = OneHotEncoder(dtype='uint16', handle_unknown="ignore")

ohe_x_train = ohe.fit_transform(x_train[ohe_features])

ohe_x_test = ohe.transform(x_test[ohe_features])
# OneHotEncoder transfer 之后的默认的是一个稀疏矩阵，可以通过 to_array 或者设置 sparse=False 等转化成 正常的 Array

type(ohe_x_train)
ohe_x_train.shape
ohe_x_train[:,0]
def encode(data, col, max_val):

    data[col + '_sin'] = np.sin(2 * np.pi * data[col]/max_val)

    data[col + '_cos'] = np.cos(2 * np.pi * data[col]/max_val)

    return data

#直接生成2列



x_train = encode(x_train, 'month', 12)

x_train = encode(x_train, 'day', 7)



x_test = encode(x_test, 'month', 12)

x_test = encode(x_test, 'day', 7)
cyclic_x_train = x_train[['month_sin','month_cos','day_sin','day_cos']]

cyclic_x_test = x_test[['month_sin','month_cos','day_sin','day_cos']]
cyclic_x_test.head(1).T
from sklearn.preprocessing import OrdinalEncoder



oe = OrdinalEncoder()

oe_x_train = oe.fit_transform(x_train[oe_features])

oe_x_test = oe.transform(x_test[oe_features])
from category_encoders import TargetEncoder

from sklearn.model_selection import StratifiedKFold



# 很高级的一种做法，做 Target Encoding 的时候拆分开来做

# oof 的含义是 out of fold

def transform(transformer, x_train, y_train, cv):

    oof = pd.DataFrame(index=x_train.index, columns=x_train.columns)

    for train_idx, valid_idx in cv.split(x_train, y_train):

        x_train_train = x_train.loc[train_idx]

        y_train_train = y_train.loc[train_idx]

        x_train_valid = x_train.loc[valid_idx]

        transformer.fit(x_train_train, y_train_train)

        oof_part = transformer.transform(x_train_valid)

        oof.loc[valid_idx] = oof_part

    return oof



target = TargetEncoder(drop_invariant=True, smoothing=0.2)



cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# shuffle：在每次划分时，是否进行洗牌

# ①若为Falses时，其效果等同于random_state等于整数，每次划分的结果相同

# ②若为True时，每次划分的结果都不一样，表示经过洗牌，随机取样的

# target_x_train 用的是 oof 技术计算的

target_x_train = transform(target, x_train[target_features], y_train, cv).astype('float')

# target_x_test 用的是 train 的全量target去计算的

target.fit(x_train[target_features], y_train)

# 用 fit 完成后的对象，去transfer x_test

target_x_test = target.transform(x_test[target_features]).astype('float')



#生成的是 DF 格式

# type(target_x_test)
target.get_params()
#生成的是 DF

type(target_x_test)
#生成的是 DF

type(target_x_train)
target_x_test.head(1).T
import scipy

x_train = scipy.sparse.hstack([ohe_x_train, simple_x_train]).tocsr()

x_test = scipy.sparse.hstack([ohe_x_test, simple_x_test]).tocsr()
x_train.shape
type(x_train)
type(ohe_x_train)
# 对 logit 做网格搜索



from sklearn.linear_model import LogisticRegression



logit_param_grid = {

    'C': list(np.linspace(start = 0, stop = 0.1, num = 11))

}



logit_grid = GridSearchCV(LogisticRegression(solver='lbfgs'), logit_param_grid,

                          scoring='roc_auc', cv=5)

logit_grid.fit(x_train, y_train)



best_C = logit_grid.best_params_['C']

best_Score = logit_grid.best_score_

print('Best C:', best_C)

print('Best Score:', best_Score)
# 用 Best C predict x_test

logit = LogisticRegression(

    C=best_C, 

    solver='lbfgs', 

    max_iter=10000)

logit.fit(x_train, y_train)

y_pred_logit = logit.predict_proba(x_test)[:, 1]
# 看在 Train Set 上的AUC结果如何

y_train_pred_logit = logit.predict_proba(x_train)[:, 1]

train_auc_logit = roc_auc_score(y_train, y_train_pred_logit)

train_auc_logit
x_train = np.concatenate((oe_x_train, simple_x_train, target_x_train, cyclic_x_train), axis=1)

x_test = np.concatenate((oe_x_test, simple_x_test, target_x_test, cyclic_x_test), axis=1)
from sklearn.model_selection import KFold

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import roc_auc_score

from xgboost import plot_importance

from sklearn.metrics import make_scorer



import xgboost as xgb



## Hyperopt modules

from hyperopt import fmin, hp, tpe, Trials, space_eval, STATUS_OK, STATUS_RUNNING

from functools import partial

import gc



import time

def objective(params):

    time1 = time.time()

    params = {

        'max_depth': int(params['max_depth']),

        'gamma': "{:.3f}".format(params['gamma']),

        'subsample': "{:.2f}".format(params['subsample']),

        'reg_alpha': "{:.3f}".format(params['reg_alpha']),

        'reg_lambda': "{:.3f}".format(params['reg_lambda']),

        'learning_rate': "{:.3f}".format(params['learning_rate']),

        'num_leaves': '{:.3f}'.format(params['num_leaves']),

        'colsample_bytree': '{:.3f}'.format(params['colsample_bytree']),

        'min_child_samples': '{:.3f}'.format(params['min_child_samples']),

        'feature_fraction': '{:.3f}'.format(params['feature_fraction']),

        'bagging_fraction': '{:.3f}'.format(params['bagging_fraction'])

    }



    print("\n############## New Run ################")

    print(f"params = {params}")

    FOLDS = 12

    count=1

    kf = StratifiedKFold(n_splits=FOLDS, shuffle=False, random_state=42)



    # tss = TimeSeriesSplit(n_splits=FOLDS)

#     y_preds = np.zeros(submission.shape[0]) #这句看起来没有用

    # y_oof = np.zeros(X_train.shape[0])

    score_mean = 0 #初始化 mean

    for tr_idx, val_idx in kf.split(x_train, y_train):

        clf = xgb.XGBClassifier(

            n_estimators=1000, random_state=4, 

            verbose=True, 

            

            tree_method='gpu_hist', # GPU加速

            **params #这个用法需要注意

        )



        X_tr, X_vl = x_train[tr_idx, :], x_train[val_idx, :] # 需要根据 x_train 的类型来判断是否用 iloc

        y_tr, y_vl = y_train.iloc[tr_idx], y_train.iloc[val_idx]

        

        clf.fit(X_tr, y_tr) #Training Set内部的K Fold # 用fit方法没有早停

        

        #y_pred_train = clf.predict_proba(X_vl)[:,1]

        #print(y_pred_train)

        

        score = make_scorer(roc_auc_score, needs_proba=True)(clf, X_vl, y_vl) # 自定义一个score，别的项目需要替换掉这个

        # plt.show()

        score_mean += score # 先把这几次Fold的mean加起来

        print(f'{count} CV - score: {round(score, 4)}')

        count += 1 # count 单纯是为了用来计数 表示是第几次cv

    time2 = time.time() - time1

    print(f"Total Time Run: {round(time2 / 60,2)}")

    gc.collect() #内存回收机制，检查是否有内存泄漏

    print(f'Mean ROC_AUC: {score_mean / FOLDS}') #得到最终的mean

    del X_tr, X_vl, y_tr, y_vl, clf, score

    

    return -(score_mean / FOLDS) #这里是由于要取损失函数的最小值，因为score前面要有负号



space = {

    # The maximum depth of a tree, same as GBM.

    # Used to control over-fitting as higher depth will allow model 

    # to learn relations very specific to a particular sample.

    # Should be tuned using CV.

    # Typical values: 3-10

    'max_depth': hp.quniform('max_depth', 2, 8, 1),

    

    # reg_alpha: L1 regularization term. L1 regularization encourages sparsity 

    # (meaning pulling weights to 0). It can be more useful when the objective

    # is logistic regression since you might need help with feature selection.

    'reg_alpha':  hp.uniform('reg_alpha', 0.01, 0.4),

    

    # reg_lambda: L2 regularization term. L2 encourages smaller weights, this

    # approach can be more useful in tree-models where zeroing 

    # features might not make much sense.

    'reg_lambda': hp.uniform('reg_lambda', 0.01, .4),

    

    # eta: Analogous to learning rate in GBM

    # Makes the model more robust by shrinking the weights on each step

    # Typical final values to be used: 0.01-0.2

    'learning_rate': hp.uniform('learning_rate', 0.01, 0.15),

    

    # colsample_bytree: Similar to max_features in GBM. Denotes the 

    # fraction of columns to be randomly samples for each tree.

    # Typical values: 0.5-1

    'colsample_bytree': hp.uniform('colsample_bytree', 0.3, 1),

    

    # A node is split only when the resulting split gives a positive

    # reduction in the loss function. Gamma specifies the 

    # minimum loss reduction required to make a split.

    # Makes the algorithm conservative. The values can vary depending on the loss function and should be tuned.

    'gamma': hp.uniform('gamma', 0.01, .7),

    

    # more increases accuracy, but may lead to overfitting.

    # num_leaves: the number of leaf nodes to use. Having a large number 

    # of leaves will improve accuracy, but will also lead to overfitting.

    'num_leaves': hp.choice('num_leaves', list(range(20, 200, 5))),

    

    # specifies the minimum samples per leaf node.

    # the minimum number of samples (data) to group into a leaf. 

    # The parameter can greatly assist with overfitting: larger sample

    # sizes per leaf will reduce overfitting (but may lead to under-fitting).

    'min_child_samples': hp.choice('min_child_samples', list(range(100, 250, 10))),

    

    # subsample: represents a fraction of the rows (observations) to be 

    # considered when building each subtree. Tianqi Chen and Carlos Guestrin

    # in their paper A Scalable Tree Boosting System recommend 

    'subsample': hp.choice('subsample', [.5, 0.6, 0.7, .8]),

    

    # randomly select a fraction of the features.

    # feature_fraction: controls the subsampling of features used

    # for training (as opposed to subsampling the actual training data in 

    # the case of bagging). Smaller fractions reduce overfitting.

    'feature_fraction': hp.uniform('feature_fraction', 0.4, .8),

    

    # randomly bag or subsample training data.

    'bagging_fraction': hp.uniform('bagging_fraction', 0.4, .9)

    

    # bagging_fraction and bagging_freq: enables bagging (subsampling) 

    # of the training data. Both values need to be set for bagging to be used.

    # The frequency controls how often (iteration) bagging is used. Smaller

    # fractions and frequencies reduce overfitting.

}
best = fmin(fn=objective,

            space=space,

            algo=tpe.suggest,

            max_evals=50, 

            # trials=trials #trials 是为了后来绘图所用，详细课件 hyperopt 教程

           )

# 整个 fmin 之后返回的是一个参数空间
best_params = space_eval(space, best) #可能是由于索引问题，公式得到的best，需要重新加工一下才是最终的 best_params

best_params['max_depth'] = int(best_params['max_depth'])

best_params
clf = xgb.XGBClassifier(

    n_estimators=500,

    **best_params,

    tree_method='gpu_hist'

)



clf.fit(x_train, y_train)



y_pred_xgb = clf.predict_proba(x_test)[:,1] 
# 看在 Train Set 上的AUC结果如何

y_train_pred_xgb = clf.predict_proba(x_train)[:, 1]

train_auc_xgb = roc_auc_score(y_train, y_train_pred_xgb)

train_auc_xgb
# Blending

y_pred = np.add(y_pred_logit, y_pred_xgb) / 2
submission = pd.read_csv('../input/cat-in-the-dat-ii/sample_submission.csv', index_col='id')

submission['target'] = y_pred

submission.to_csv('logit_xgboost.csv')
submission.head()