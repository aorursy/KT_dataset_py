# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn.metrics import roc_auc_score

from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold



import gc



import warnings

warnings.filterwarnings('ignore')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/train.csv')

test = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/test.csv')

sample_submission = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/sample_submission.csv')

train.shape, test.shape, sample_submission.shape
import lightgbm as lgb
def kfold_lightgbm(train, test, target_col, params, cols_to_drop=None, cat_features=None,num_folds=10, stratified = False, 

                   debug= False):

    

    print("Starting LightGBM. Train shape: {}, test shape: {}".format(train.shape, test.shape))





    

    # Cross validation model

    if stratified:

        folds = StratifiedKFold(n_splits= num_folds, shuffle=True, random_state=1)

    else:

        folds = KFold(n_splits= num_folds, shuffle=True, random_state=1)



    # Create arrays and dataframes to store results

    oof_preds = np.zeros(train.shape[0])

    sub_preds = np.zeros(test.shape[0])

    feature_importance_df = pd.DataFrame()

    if cols_to_drop == None:

        feats = [f for f in train.columns if f not in [target_col]]

    else:

        feats = [f for f in train.columns if f not in cols_to_drop+[target_col]]



    # k-fold

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train[feats], train[target_col])):

        train_x, train_y = train[feats].iloc[train_idx], train[target_col].iloc[train_idx]

        valid_x, valid_y = train[feats].iloc[valid_idx], train[target_col].iloc[valid_idx]



        # set data structure

        lgb_train = lgb.Dataset(train_x,

                                label=train_y,

                                categorical_feature=cat_features,

                                free_raw_data=False)

        lgb_test = lgb.Dataset(valid_x,

                               label=valid_y,

                               categorical_feature=cat_features,

                               free_raw_data=False)



        # params after optimization

        reg = lgb.train(

                        params,

                        lgb_train,

                        valid_sets=[lgb_train, lgb_test],

                        valid_names=['train', 'test'],

                        )



        roc_auc = []

        oof_preds[valid_idx] = reg.predict(valid_x, num_iteration=reg.best_iteration)

        sub_preds += reg.predict(test[feats], num_iteration=reg.best_iteration) / folds.n_splits



        fold_importance_df = pd.DataFrame()

        fold_importance_df["feature"] = feats

        fold_importance_df["importance"] = np.log1p(reg.feature_importance(importance_type='gain', 

                                                                           iteration=reg.best_iteration))

        fold_importance_df["fold"] = n_fold + 1

        

        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

        print('Fold %2d ROC-AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx])))

        roc_auc.append(roc_auc_score(valid_y, oof_preds[valid_idx]))

        del reg, train_x, train_y, valid_x, valid_y

        gc.collect()

        

    print('Mean ROC-AUC : %.6f' % (np.mean(roc_auc)))

    return sub_preds



def make_submit(y_pred, filename):

    submit_df = pd.DataFrame(y_pred, columns=['target'], index=sample_submission['id'])

    submit_df.to_csv(f'{filename}')

    print(f'Done. Commit solution and then upload {filename} file')

    submit_df['target'].hist()
cat_features=[x for x in range(25)]

cat_features_names = test.columns



params ={

    'objective': 'binary',

    'metric': 'roc_auc',

    'categorical_features': cat_features

                }
# As lightGBM only accepts number as categorical features have to use LabelEncoder

from sklearn.preprocessing import LabelEncoder



le = LabelEncoder()



for col in cat_features_names:

    train[col] = train[col].astype('str')

    test[col] = test[col].astype('str')

    

    train[col] = le.fit_transform(train[col])

    test[col] = le.fit_transform(test[col])
y_pred_baseline = kfold_lightgbm(train, test, 'target', cat_features=cat_features, params=params)
make_submit(y_pred_baseline, 'baseline_submit.csv')