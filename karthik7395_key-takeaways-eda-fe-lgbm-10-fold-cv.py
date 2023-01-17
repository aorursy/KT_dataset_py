# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")

print("Number of rows and columns in train set : ",train_df.shape)

print("Number of rows and columns in test set : ",test_df.shape)
train_df.head()
train_df.target.value_counts(dropna=False, normalize=True).plot(kind="bar")
corr_df=train_df.iloc[:,1:].corr()
import seaborn as sns

sns.set(style="white")

mask = np.zeros_like(corr_df.iloc[:,1:], dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr_df.iloc[:,1:], mask=mask, cmap=cmap, vmax=.2, center=0,

            square=True, linewidths=.5)
corr_df.loc[corr_df.target>0.05].index[1:]
corr_target=corr_df.loc[corr_df.target>0.05]['target'].iloc[1:]

corr_target.plot(kind='bar')
train_df.iloc[:,2:].describe()
train_df_out_rem=train_df.iloc[:,1:]
train_df_out_rem.shape
train_df_out_rem=train_df_out_rem.reset_index(drop=True)
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, max_depth=8,random_state=0)
clf.fit(train_df_out_rem.iloc[:,1:], train_df_out_rem.iloc[:,0])
importance_dict=dict(zip(list(train_df_out_rem.columns[1:]),list(clf.feature_importances_)))
importance_dict_imp={k: v for k, v in importance_dict.items() if v >0.01}
features=list(importance_dict_imp.keys())

indices = np.argsort(list(importance_dict_imp.values()))

features_sorted=list(importance_dict_imp.values())

features_sorted_upd=[features_sorted[i] for i in indices]

features_names_upd=[features[i] for i in indices]

plt.figure(figsize = (6,12))

plt.title('Feature Importances')

plt.barh(features_names_upd, features_sorted_upd, color='b', align='center')
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(interaction_only=True)
poly_features=poly.fit_transform(train_df_out_rem[features_names_upd])
poly_features_df=pd.DataFrame(poly_features)

poly_features_df.columns=['Imp_feature_'+str(i) for i in range(poly_features.shape[1])]

poly_features_df.head()
other_cols=train_df_out_rem.loc[:, ~train_df_out_rem.columns.isin(features_names_upd)]
other_cols.shape,poly_features_df.shape
train_final_df=pd.concat([poly_features_df,other_cols],axis=1)
train_final_df.target.value_counts(dropna=False)
train_x=train_final_df.loc[:, train_final_df.columns != 'target']

train_y=train_final_df.target
train_x.shape,train_y.shape
poly_features_test=poly.transform(test_df[features_names_upd])
poly_features_test.shape
poly_features_test_df=pd.DataFrame(poly_features_test)

poly_features_test_df.columns=['Imp_feature_'+str(i) for i in range(poly_features_test.shape[1])]

poly_features_test_df.head()
other_cols_test=test_df.loc[:, ~test_df.columns.isin(features_names_upd)]
id_code=other_cols_test.ID_code
other_cols_test=other_cols_test.drop(columns=['ID_code'])
final_test_df=pd.concat([poly_features_test_df,other_cols_test],axis=1)
final_test_df.head()
# reduce memory

def reduce_mem_usage(df, verbose=True):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage().sum() / 1024**2

    for col in df.columns:

        col_type = df[col].dtypes

        if col_type in numerics:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)



    end_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))

    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))



    return df
train_x=reduce_mem_usage(train_x)

final_test_df=reduce_mem_usage(final_test_df)
import gc

gc.collect()

del train_df,test_df,train_df_out_rem
from sklearn.metrics import roc_auc_score 

from sklearn.model_selection import KFold, StratifiedKFold

import lightgbm as lgb
def kfold_lightgbm(train_df,target,test_df, num_folds, stratified = True, debug= False):

    print("Starting LightGBM. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))



    # Cross validation model

    if stratified:

        folds = StratifiedKFold(n_splits= num_folds, shuffle=True, random_state=326)

    else:

        folds = KFold(n_splits= num_folds, shuffle=True, random_state=326)



    # Create arrays and dataframes to store results

    oof_preds = np.zeros(train_df.shape[0])

    sub_preds = np.zeros(test_df.shape[0])

    feature_importance_df = pd.DataFrame()



    # k-fold

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df,target)):

        train_x, train_y = train_df.iloc[train_idx], target[train_idx]

        valid_x, valid_y = train_df.iloc[valid_idx], target[valid_idx]



        # set data structure

        lgb_train = lgb.Dataset(train_x,

                                label=train_y,

                                free_raw_data=False)

        lgb_test = lgb.Dataset(valid_x,

                               label=valid_y,

                               free_raw_data=False)



        # params optimized by optuna

        params  = {

        'num_leaves': 20,

        'max_bin': 60,

        'min_data_in_leaf': 17,

        'learning_rate': 0.07,

        'min_sum_hessian_in_leaf': 0.000446,

        'bagging_fraction': 1.0, 

        'bagging_freq': 5, 

        'lambda_l1': 4,

        'lambda_l2': 1,

        'min_gain_to_split': 0.15,

        'max_depth': 18,

        'save_binary': True,

        'objective': 'binary',

        'boosting_type': 'gbdt',

        'verbose': 1,

        'metric': 'auc',

        'is_unbalance': True,

    }



        reg = lgb.train(

                        params,

                        lgb_train,

                        valid_sets=[lgb_test],

                        valid_names=[ 'test'],

                        num_boost_round=10000,

                        early_stopping_rounds= 200,

                        verbose_eval=100

                        )



        oof_preds[valid_idx] = reg.predict(valid_x, num_iteration=reg.best_iteration)

        sub_preds += reg.predict(test_df, num_iteration=reg.best_iteration) / folds.n_splits



        fold_importance_df = pd.DataFrame()

        fold_importance_df["feature"] = list(test_df.columns)

        fold_importance_df["importance"] = np.log1p(reg.feature_importance(importance_type='gain', iteration=reg.best_iteration))

        fold_importance_df["fold"] = n_fold + 1

        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx])))

        del reg, train_x, train_y, valid_x, valid_y

        gc.collect()

    return sub_preds,oof_preds
test_preds,train_preds=kfold_lightgbm(train_x,train_y.values,final_test_df, 10)
roc_auc_score(train_y.values,train_preds)
subm=pd.read_csv('../input/sample_submission.csv')
subm.target=test_preds

subm.to_csv('submission.csv',index=False)