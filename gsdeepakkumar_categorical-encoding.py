# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

from tqdm import tqdm

import random



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
## read the data:

kaggle=1



if kaggle==1:

    

    train=pd.read_csv("/kaggle/input/cat-in-the-dat/train.csv")

    test=pd.read_csv("/kaggle/input/cat-in-the-dat/test.csv")

    sample=pd.read_csv("/kaggle/input/cat-in-the-dat/sample_submission.csv")

    

else:

    train=pd.read_csv("train.csv")

    test=pd.read_csv("test.csv")

    sample=pd.read_csv("sample_submission.csv")
train.head()
train.dtypes
train.isnull().any(),test.isnull().any()
for c in train.columns :

    if c not in ['id','target']:

        print(c,"has",train[c].nunique(),"unique values\n")
for col in ['nom_5','nom_6','nom_7','nom_8','nom_9']:

    print(col,"has ",set(train[col].unique())-set(test[col].unique())," unique value in train set not available in test dataset\n")
train.shape,test.shape
train.columns,test.columns
train.dtypes,test.dtypes
def reduce_memory(df,col):

    mx = df[col].max()

    print(f'\n reducing memory for {col}')

    if mx<256:

        df[col] = df[col].astype('uint8')

    elif mx<65536:

        df[col] = df[col].astype('uint16')

    else:

        df[col] = df[col].astype('uint32')






def reduce_cardi(train,test,col):

    print(f'\nBefore reducing the cardinality of col {c} in train was {train[c].nunique()}')

    cv1=pd.DataFrame(train[col].value_counts().reset_index().rename({col:'train'},axis=1))

    cv2=pd.DataFrame(test[col].value_counts().reset_index().rename({col:'test'},axis=1))

    cv3=pd.merge(cv1,cv2,on='index',how='outer')

    factor=len(train)/len(test)

    cv3['train'].fillna(0,inplace=True)

    cv3['test'].fillna(0,inplace=True)

    cv3['remove']=False

    cv3['remove']=cv3['remove']|(cv3['train']<len(train)/10000)  ## remove variables that appear 0.1 % of the total data in train.

    cv3['new'] = cv3.apply(lambda x: x['index'] if x['remove']==False else 0,axis=1)

    cv3['new'],_ = cv3['new'].factorize(sort=True)

    cv3.set_index('index',inplace=True)

    cc = cv3['new'].to_dict()

    train[col] = train[col].map(cc)

    

    test[col]=test[col].map(cc)

    #reduce_memory(test,col)
card_cols=['nom_9','nom_8','nom_5','nom_6','nom_7']



for c in card_cols:

    print(f'\nReducing cardinality for col {c}')

    reduce_cardi(train,test,c)

    reduce_memory(train,c)

    reduce_memory(test,c)

    print(f'\nAfter reducing the cardinality of col {c} in train is {train[c].nunique()}')
## Creating new features after doing one round of training and understanding the gini importance



# train['ord_1'+'_'+'ord_5']=train['ord_1']+"_"+train['ord_5']

# test['ord_1'+'_'+'ord_5']=test['ord_1']+"_"+test['ord_5']



# train['ord_1'+'_'+'ord_2']=train['ord_1']+"_"+train['ord_2']

# test['ord_1'+'_'+'ord_2']=test['ord_1']+"_"+test['ord_2']



# train['nom_6'+'_'+'nom_7']=train['nom_6']+'_'+train['nom_7']

# test['nom_6'+'_'+'nom_7']=test['nom_6']+'_'+test['nom_7']
# a=train

# b=test
## OHE columns with less cardinality



cat_cols=['bin_0','bin_1','bin_2','bin_3','bin_4','ord_0','ord_1','ord_2']

#https://github.com/rushter/heamy/blob/c330854cee3c547417eb353a4a4a23331b40b4bc/heamy/feature.py



for column in cat_cols:

      

        cate = pd.concat([train[column], test[column]]).dropna().unique()



        train[column] = train[column].astype('category')

        test[column] = test[column].astype('category')



train = pd.get_dummies(train, columns=cat_cols, dummy_na=False, sparse=False)

test = pd.get_dummies(test, columns=cat_cols, dummy_na=False, sparse=False)

train.columns,test.columns
import category_encoders as ce
def frequency_encoding(variable):

    t = train[variable].value_counts().reset_index()

    t = t.reset_index()

    t.loc[t[variable] == 1, 'level_0'] = np.nan

    t.set_index('index', inplace=True)

    max_label = t['level_0'].max() + 1

    t.fillna(max_label, inplace=True)

    return t.to_dict()['level_0']
# temp = test['nom_6_nom_7'].value_counts().reset_index()

# temp = temp.reset_index()

# temp.loc[temp['nom_6_nom_7'] == 1, 'level_0'] = np.nan

# temp.set_index('index', inplace=True)

# max_label = temp['level_0'].max() + 1

# temp.fillna(max_label, inplace=True)

# #temp.to_dict()['level_0'],max_label

# temp.isna().any()
ce_cols=['nom_0','nom_1','nom_2','nom_3','nom_4', 'ord_3', 'ord_4', 'ord_5', 'day', 'month']

## Remove day,month and newly created combined columns:

#ce_cols=['nom_0','nom_1','nom_2','nom_3','nom_4','nom_5','nom_6','nom_7','nom_8', 'nom_9', 'ord_3', 'ord_4', 'ord_5']

## Remove cols that have been summarised earlier:

#ce_cols = [c for c in ce_cols if c not in eda_cols]
ce_cols
# for c in ce_cols:

#     print(f'\nConverting {c} to lower case')

#     train[c]=train[c].str.lower()

#     test[c]=test[c].str.lower()
# cat_boost=ce.CatBoostEncoder(cols=ce_cols)

# cat_boost.fit(train[ce_cols],train['target'])



# train=train.join(cat_boost.transform(train[ce_cols]).add_suffix('_cb'))

# test=test.join(cat_boost.transform(test[ce_cols]).add_suffix('_cb'))
## Frequency encode variables:

for variable in tqdm(ce_cols):

    freq_encod_dict=frequency_encoding(variable)

    train[variable+'_FE']=train[variable].map(lambda x:freq_encod_dict.get(x,1))# return value as 1 if the variable does not exist in freq_encod_dict

    test[variable+'_FE']=test[variable].map(lambda x:freq_encod_dict.get(x,1))

    #categorical_columns.remove(variable)
train.columns,test.columns
# test.isnull().any()
train.head()
ce_cols
train=train.drop(columns=ce_cols,axis=1)

test=test.drop(columns=ce_cols,axis=1)
# train=train.drop(columns=['nom_5','nom_6','nom_7','nom_8','ord_5'],axis=1)

# test=test.drop(columns=['nom_5','nom_6','nom_7','nom_8','ord_5'],axis=1)
# plt.figure(figsize=(30,30))

# sns.heatmap(train[feats].corr(), cmap='RdBu_r', annot=True, center=0.0)
##https://www.kaggle.com/kyakovlev/ieee-fe-for-local-test

def seed_everything(seed=0):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)
seed_everything(1001)
##import required libraries:

from sklearn.model_selection import StratifiedKFold,GroupKFold

import lightgbm as lgb

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import roc_auc_score,roc_curve

from sklearn.linear_model import LogisticRegression

from bayes_opt import BayesianOptimization

import gc

from tqdm import tqdm

#import lofo as lofo
## Split into train and validation:

y=train['target']

train.drop('target',axis=1,inplace=True)
n_folds=5

kf=StratifiedKFold(n_splits=n_folds,shuffle=True,random_state=1001)

feats=[f for f in train.columns if f not in ['id','bin_0_1',

 'bin_1_1',

 'bin_2_0',

 'bin_3_F',

 'bin_4_N', 'ord_0_1','ord_1_Novice','ord_2_Lava Hot' ]]  # remove highly correlated features
oof_preds=np.zeros(train.shape[0])

sub_preds=np.zeros(test.shape[0])



feature_importance_df=pd.DataFrame()

categorical_features=[f for f in train.columns if f not in ['id','bin_0_1',

 'bin_1_1',

 'bin_2_0',

 'bin_3_F',

 'bin_4_N','ord_0_1','ord_1_Novice','ord_2_Lava Hot']]
categorical_features
##https://www.kaggle.com/cdeotte/eda-for-columns-v-and-id

plt.figure(figsize=(30,30))

sns.heatmap(train[feats].corr(), cmap='RdBu_r', annot=True, center=0.0)
cor=train[feats].corr()

##https://www.kaggle.com/gpreda/santander-eda-and-prediction

c=cor.unstack()

so=c.sort_values(kind='quicksort').reset_index().rename({0:'value'},axis=1)

so=so[so['level_0']!=so['level_1']]

print("Negative correlated features \n",so.head(10),"\nPositive correlated features\n",so.tail(10))
#Parameters through Bayesian Optimization:



param = {'num_leaves': 40,

         'min_data_in_leaf': 69, 

         'objective':'binary',

         'max_depth': 4,

         'learning_rate': 0.026,

         "boosting": "gbdt",

         "feature_fraction": 0.49,

         "metric": 'auc',

         "lambda_l2": 2.84,

         "random_state": 100,

         "min_gain_to_split":0.386,

         "bagging_freq":5, ## randomly initialized

         "bagging_fraction":0.5,## randomly initialized

         "verbosity": -1}



# {'feature_fraction': 0.49667591307631404,

#  'lambda_l2': 2.843966944905738,

#  'learning_rate': 0.026747986947564972,

#  'max_depth': 4.390096819896572,

#  'min_data_in_leaf': 68.92767604826321,

#  'min_gain_to_split': 0.38669168478099314,

#  'num_leaves': 39.828939074908384}



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
# ## Feature importance:



# dataset=lofo.Dataset(df=train,target="target",features=categorical_features)



# lgbm=lgb.train(param,num_round,

#                     verbose_eval=100,

#                     early_stopping_rounds = 200)



# lf=LOFOImportance(dataset,model=lgbm,cv=4,scoring='roc_auc',n_jobs=4)



# # dataset = Dataset(df=df, target="binary_target", features=features, feature_groups=feature_groups)



# # lgbm = LGBMClassifier(random_state=0, n_jobs=1)



# # lofo = LOFOImportance(dataset, model=lgbm, cv=4, scoring='roc_auc', n_jobs=4)
## Using the standard cross validation format for training.





for n_folds,(train_idx,valid_idx) in tqdm(enumerate(kf.split(train.values,y.values))):

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

    oof_preds[valid_idx]=clf.predict(train.iloc[valid_idx][feats],num_iteration=clf.best_iteration)

    sub_preds+=clf.predict(test[feats],num_iteration=clf.best_iteration)/kf.n_splits

    

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

# ##https://www.kaggle.com/cdeotte/xgb-fraud-with-magic-0-9600

# kf=GroupKFold(n_splits=12)
# ## Using the standard cross validation format for training.





# for n_folds,(train_idx,valid_idx) in tqdm(enumerate(kf.split(train.values,y.values,groups=train['month']))):

    

#     print("fold n°{}".format(n_folds+1))

#     month = train.iloc[valid_idx]['month'].iloc[0]

#     print('\nFold',n_folds+1,'withholding month',month)

#     print('\n rows of train =',len(train_idx),'rows of holdout =',len(valid_idx))

#     trn_data = lgb.Dataset(train.iloc[train_idx][feats],

#                            label=y.iloc[train_idx],

#                            categorical_feature=categorical_features

#                           )

#     val_data = lgb.Dataset(train.iloc[valid_idx][feats],

#                            label=y.iloc[valid_idx],categorical_feature=categorical_features

#                           )



#     num_round = 10000

#     clf = lgb.train(param,

#                     trn_data,

#                     num_round,

#                     valid_sets = [trn_data, val_data],

#                     verbose_eval=100,

#                     early_stopping_rounds = 200)

#     oof_preds[valid_idx]=clf.predict(train.iloc[valid_idx][feats],num_iteration=clf.best_iteration)

#     sub_preds+=clf.predict(test[feats],num_iteration=clf.best_iteration)/kf.n_splits

    

#     fold_importance_df=pd.DataFrame()

#     fold_importance_df['features']=feats

#     fold_importance_df['importance']=clf.feature_importance(importance_type='gain')

#     fold_importance_df['folds']=n_folds+1

#     print(f'Fold {n_folds+1}: Most important features are:\n')

#     for i in np.argsort(fold_importance_df['importance'])[-5:]:

#         print(f'{fold_importance_df.iloc[i,0]}-->{fold_importance_df.iloc[i,1]}')

    

#     feature_importance_df=pd.concat([feature_importance_df,fold_importance_df],axis=0)

    

#     print('Fold %2d AUC : %.6f' % (n_folds + 1, roc_auc_score(y.iloc[valid_idx], oof_preds[valid_idx])))

#     del clf

#     gc.collect()

    





# print('Full auc score %.6f' % (roc_auc_score(y,oof_preds)))



# test['target']=sub_preds

# ## Random Forest

# for n_folds,(train_idx,valid_idx) in tqdm(enumerate(kf.split(train.values,y.values))):

#     print("fold n°{}".format(n_folds+1))

#     trn_X = train.iloc[train_idx][feats]

#     trn_Y = y.iloc[train_idx]

#     val_X=train.iloc[valid_idx][feats]

#     val_Y=y.iloc[valid_idx]

#     num_round = 10000

#     clf =RandomForestClassifier(n_estimators=50,min_samples_split=10,min_samples_leaf=10,max_depth=10,random_state=100,criterion='gini',max_features='sqrt',oob_score=True)

#     clf.fit(trn_X,trn_Y)

#     oof_preds[valid_idx]=clf.predict(train.iloc[valid_idx][feats])

#     sub_preds+=clf.predict(test[feats])/kf.n_splits

    

#     fold_importance_df=pd.DataFrame()

#     fold_importance_df['features']=feats

#     fold_importance_df['importance']=clf.feature_importances_

#     fold_importance_df['folds']=n_folds+1

#     print(f'Fold {n_folds+1}: Most important features are:\n')

#     for i in np.argsort(fold_importance_df['importance'])[-5:]:

#         print(f'{fold_importance_df.iloc[i,0]}-->{fold_importance_df.iloc[i,1]}')

    

#     feature_importance_df=pd.concat([feature_importance_df,fold_importance_df],axis=0)

    

#     print('Fold %2d AUC : %.6f' % (n_folds + 1, roc_auc_score(y.iloc[valid_idx], oof_preds[valid_idx])))

#     del clf

#     gc.collect()

    





# print('Full auc score %.6f' % (roc_auc_score(y,oof_preds)))



# test['target']=sub_preds
sample['target']=sub_preds
sample.head()
sample.to_csv("sample_submission.csv",index=False)