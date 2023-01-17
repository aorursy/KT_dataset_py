#https://www.kaggle.com/cdeotte/high-scoring-lgbm-malware-0-702-0-775

#https://www.kaggle.com/fabiendaniel/detecting-malwares-with-lgbm

#https://www.kaggle.com/humananalog/xgboost-lasso

#https://www.kaggle.com/ogrellier/good-fun-with-ligthgbm

#https://www.kaggle.com/mlisovyi/modular-good-fun-with-ligthgbm/output




import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns





import gc

from tqdm import tqdm
kaggle=1



if kaggle==0:

    train=pd.read_csv("train.csv")

    test=pd.read_csv("test.csv")

    sample_submission=pd.read_csv("sample_submission.csv")

    

else:

    train=pd.read_csv("../input/cat-in-the-dat/train.csv")

    test=pd.read_csv("../input/cat-in-the-dat/test.csv")

    sample_submission=pd.read_csv("../input/cat-in-the-dat/sample_submission.csv")
train.head()
train.shape,test.shape
train['target'].value_counts()
train.dtypes
test.dtypes
#convert all the columns to category datatype:

for f in train.columns:

    if f=="id" or f=="target": continue

    print(f'Converting {f} into category datatype\n')

    train[f]=train[f].astype('category')

    test[f]=test[f].astype('category')
## For binary columns , the cardinality will be 2.Lets separate them out .

binary_columns=[c for c in train.columns if train[c].nunique()==2]
binary_columns
categorical_columns=[c for c in train.columns if (c not in binary_columns)]
cardinality=[]

for c in categorical_columns:

    if c=='id':continue

    cardinality.append([c,train[c].nunique()])

cardinality.sort(key=lambda x:x[1],reverse=True)

cardinality
# Columns that can be safely label encoded

good_label_cols = [col for col in categorical_columns if 

                   set(train[col]) == set(test[col])]
## from https://www.kaggle.com/fabiendaniel/detecting-malwares-with-lgbm

def frequency_encoding(variable):

    t = pd.concat([train[variable], test[variable]]).value_counts().reset_index()

    t = t.reset_index()

    t.loc[t[variable] == 1, 'level_0'] = np.nan

    t.set_index('index', inplace=True)

    max_label = t['level_0'].max() + 1

    t.fillna(max_label, inplace=True)

    return t.to_dict()['level_0']
#frequency_encoded_columns=['nom_9','nom_8','nom_7','nom_6','nom_5','ord_5','ord_4']
for variable in tqdm(good_label_cols):

    freq_encod_dict=frequency_encoding(variable)

    train[variable+'_FE']=train[variable].map(lambda x:freq_encod_dict.get(x,np.nan))

    test[variable+'_FE']=test[variable].map(lambda x:freq_encod_dict.get(x,np.nan))

    categorical_columns.remove(variable)
#https://www.kaggle.com/vprokopev/mean-likelihood-encodings-a-comprehensive-study



def factorize(train, test, features, na_value=-9999, full=False, sort=True):

    """Factorize categorical features.

    Parameters

    ----------

    train : pd.DataFrame

    test : pd.DataFrame

    features : list

           Column names in the DataFrame to be encoded.

    na_value : int, default -9999

    full : bool, default False

        Whether use all columns from train/test or only from train.

    sort : bool, default True

        Sort by values.

    Returns

    -------

    train : pd.DataFrame

    test : pd.DataFrame

    """



    for column in features:

        if full:

            vs = pd.concat([train[column], test[column]])

            labels, indexer = pd.factorize(vs, sort=sort)

        else:

            labels, indexer = pd.factorize(train[column], sort=sort)



        train[column+'_LE'] = indexer.get_indexer(train[column])

        test[column+'_LE'] = indexer.get_indexer(test[column])



        if na_value != -1:

            train[column] = train[column].replace(-1, na_value)

            test[column] = test[column].replace(-1, na_value)



    return train, test
# indexer = {}

# for col in tqdm(categorical_columns):

#     if col == 'id': continue

#     _, indexer[col] = pd.factorize([train[col],test[col]])
#categorical_columns.remove('id')

train,test=factorize(train,test,categorical_columns,full=True)
#train,test=factorize(train,test,frequency_encoded_columns,full=True)
# for col in tqdm(categorical_columns):

#     if col=='id':continue

#     train[col+'_LE']=indexer[col].get_indexer(train[col])

#     test[col+'_LE']=indexer[col].get_indexer(test[col])

    
binary_columns
train_cat_dum=pd.DataFrame()

test_cat_dum=pd.DataFrame()

for c_ in binary_columns:

    if c_=='target':continue

    train_cat_dum=pd.concat([train_cat_dum,pd.get_dummies(train[c_],prefix=c_).astype(np.uint8)],axis=1)

    test_cat_dum=pd.concat([test_cat_dum,pd.get_dummies(test[c_],prefix=c_).astype(np.uint8)],axis=1)
train_cat_dum.head()
train=pd.concat([train,train_cat_dum],axis=1)

test=pd.concat([test,test_cat_dum],axis=1)
train.head()
train.columns,test.columns
cols_to_remove=['id', 'bin_0', 'bin_1', 'bin_2', 'bin_3', 'bin_4', 'nom_0', 'nom_1',

       'nom_2', 'nom_3', 'nom_4', 'nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9',

       'ord_0', 'ord_1', 'ord_2', 'ord_3', 'ord_4', 'ord_5', 'day', 'month','id_LE']
train=train.drop(cols_to_remove,axis=1)

test=test.drop(cols_to_remove,axis=1)
train.shape
test.shape
## Importing required libraries:

from sklearn.model_selection import KFold, StratifiedKFold

import lightgbm as lgb

from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve, average_precision_score
y=train['target']

del train['target']
n_folds=5
folds=StratifiedKFold(n_splits=5,shuffle=True,random_state=1234)

feats=[f for f in train.columns if f not in ['id']]
oof_preds = np.zeros(train.shape[0])

sub_preds = np.zeros(test.shape[0])

    

feature_importance_df = pd.DataFrame()

categorical_features=[c for c in train.columns if c not in ['id_LE']]
# param = {'num_leaves': 60,

#          'min_data_in_leaf': 60, 

#          'objective':'binary',

#          'max_depth': -1,

#          'learning_rate': 0.1,

#          "boosting": "gbdt",

#          "feature_fraction": 0.8,

#          "bagging_freq": 1,

#          "bagging_fraction": 0.8 ,

#          "bagging_seed": 11,

#          "metric": 'auc',

#          "lambda_l1": 0.1,

#          "random_state": 133,

#          "verbosity": -1}
#params after bayesian optimisation:



param = {'num_leaves': 31,

         'min_data_in_leaf': 69, 

         'objective':'binary',

         'max_depth': 4,

         'learning_rate': 0.06,

         "boosting": "gbdt",

         "feature_fraction": 0.33,

         "metric": 'auc',

         "lambda_l1": 0.01,

         "random_state": 133,

         "verbosity": -1}
for n_folds,(train_idx,valid_idx) in enumerate(folds.split(train.values,y.values)):

    print("fold n°{}".format(n_folds+1))

    trn_data = lgb.Dataset(train.iloc[train_idx][feats],

                           label=y.iloc[train_idx],

                           categorical_feature=categorical_features

                          )

    val_data = lgb.Dataset(train.iloc[valid_idx][feats],

                           label=y.iloc[valid_idx],categorical_feature=categorical_features

                          )



    num_round = 10000

    clf = lgb.train(param,

                    trn_data,

                    num_round,

                    valid_sets = [trn_data, val_data],

                    verbose_eval=100,

                    early_stopping_rounds = 200)

    

    #clf.fit(train_x,train_y,eval_set=[(train_x,train_y),(valid_x,valid_y)],verbose=500,eval_metric="auc",early_stopping_rounds=100)

    

    oof_preds[valid_idx]=clf.predict(train.iloc[valid_idx][feats],num_iteration=clf.best_iteration)

    sub_preds+=clf.predict(test[feats],num_iteration=clf.best_iteration)/folds.n_splits

    

    fold_importance_df=pd.DataFrame()

    fold_importance_df['features']=feats

    fold_importance_df['importance']=clf.feature_importance(importance_type='gain')

    fold_importance_df['folds']=n_folds+1

    print(f'Fold {n_folds+1}: Most important features are:\n')

    for i in np.argsort(fold_importance_df['importance'])[-5:]:

        print(f'{fold_importance_df.iloc[i,0]}-->{fold_importance_df.iloc[i,1]}')

    

    feature_importance_df=pd.concat([feature_importance_df,fold_importance_df],axis=0)

    

    print('Fold %2d AUC : %.6f' % (n_folds + 1, roc_auc_score(y.iloc[valid_idx], oof_preds[valid_idx])))

    del clf

    gc.collect()

    





print('Full auc score %.6f' % (roc_auc_score(y,oof_preds)))



test['target']=sub_preds

              
sample_submission['target']=sub_preds
sample_submission.head()
sample_submission.to_csv("sample_submission.csv",index=False)