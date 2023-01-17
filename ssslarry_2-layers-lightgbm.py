# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



import warnings

import math

import lightgbm as lgb

import seaborn as sns

import matplotlib.pyplot as plt

from copy import copy

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import roc_auc_score, roc_curve

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold, RepeatedKFold,StratifiedKFold

from pandas.api.types import is_numeric_dtype

from pandas.api.types import is_string_dtype

warnings.filterwarnings('ignore')

# Any results you write to the current directory are saved as output.


def get_bps_from_tree(tree_obj):

    # Find valid node index

    idx = np.arange(0, tree_obj.tree_.feature.shape[0])

    idx = list(filter(lambda x: tree_obj.tree_.feature[x] == 0, idx))

    # Find corresponding threshold for the split

    bps = sorted(tree_obj.tree_.threshold[idx])

    return bps



def discretize_feature2(df, var_col, label_col,nn,

                       max_depth=10,

                       max_leaf_nodes=5,

                       min_samples_leaf=0.05):

    tree = DecisionTreeClassifier(max_depth=max_depth,

                                  max_leaf_nodes=nn,

                                  min_samples_leaf=min_samples_leaf)

    tree = tree.fit(df[[var_col]].as_matrix(), df[label_col].values)

    bps = get_bps_from_tree(tree)

    return bps



def baddist(df,y,var,nn=2,bins=None):

    if bins is None:

        df['Bin']=df[var]

    else:

        df['Bin']=pd.cut(df[var],bins).astype('str')

    if is_string_dtype(df[var]):

        df.loc[df['Bin'].isnull(),'Bin']="null"

    if len(df[var].unique())>nn and is_numeric_dtype(df[var]) :

        if bins is None:

            bb=discretize_feature2(df=df, var_col=var, label_col=y,nn=nn)

            bb=[math.floor(f*10000)/10000 for f in bb]

            bins=[float('-inf')]+bb+[float('Inf')]

        df['Bin']=pd.cut(df[var],bins,duplicates='drop').astype('str')

    tab=pd.crosstab(df['Bin'],df[y])

    tab=tab.reset_index()

    try:

        tab=tab.sort_values(by=['Bin'])

        tab['min']=tab['Bin'].apply(lambda x: x.split(',')[0]).apply(lambda x: x[1:]).astype('float')

        tab['max']=tab['Bin'].apply(lambda x: x.split(',')[1]).apply(lambda x: x[:-1]).astype('float')

        tab=tab.sort_values('max')

        del tab['min'],tab['max']

    except: 

        tab=tab

    del df['Bin']

    tab['0%']=round(tab[0]/tab[0].sum(),4)*100

    tab['1%']=round(tab[1]/tab[1].sum(),4)*100

    tab['All']=tab[0]+tab[1]

    tab['All%']=round(tab['All']/tab['All'].sum(),4)*100

    tab['Bad Rate%']=round(tab[1]/tab['All'],4)*100

    tab['IV']=np.log(tab['1%']/tab['0%'])*(tab['1%']-tab['0%'])

    

    print(var+' IV after binning：',round(tab['IV'].sum(),4),"%")

    ivdict={"Var":[var],"IV":[round(tab['IV'].sum(),6)]}

    ivdf=pd.DataFrame.from_dict(ivdict)

    return tab,df,ivdf



def lgbmodelling(features,train_df,test_df,param=None):

    print(len(features))

    if param is None:

        param ={'bagging_fraction': 0.5,

                 'bagging_freq': 1,

                 'bagging_seed': 1993,

                 'feature_fraction': 0.05,

                 'learning_rate': 0.025,

                 'metric': 'auc',

                 'min_data_in_leaf': 50,

                 'num_leaves': 3,

                 'objective': 'binary'}

    

    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1993)

    target = train_df['target']

    print(len(target))

    oof = np.zeros(len(train_df))

    predictions = np.zeros(len(test_df))

    

    for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_df.values, target.values)):

        print("Fold {}".format(fold_+1))

        trn_data = lgb.Dataset(train_df.iloc[trn_idx][features], label=target.iloc[trn_idx])

        val_data = lgb.Dataset(train_df.iloc[val_idx][features], label=target.iloc[val_idx])



        clf = lgb.train(param, trn_data, num_boost_round=500000, valid_sets = [trn_data, val_data], verbose_eval=5000, 

                        early_stopping_rounds = 3000)

        oof[val_idx] = clf.predict(train_df.iloc[val_idx][features], num_iteration=clf.best_iteration)    

        predictions += clf.predict(test_df[features], num_iteration=clf.best_iteration) / folds.n_splits

    print("CV score: {:<8.6f}".format(roc_auc_score(target, oof)))

    return oof,predictions



train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')
features = ['var_'+str(c) for c in range(0,200)]

ivdf=pd.DataFrame()

for f in features:

    _,_,iv=baddist(df=train_df, y='target',var=f,nn=10)

    ivdf=pd.concat([ivdf,iv])

ivdf=ivdf.sort_values('IV',ascending=False)

ivdf
plt.figure(figsize=(14,28))

sns.barplot(x="IV", y="Var", data=ivdf)

plt.title('IV Ranking (of 10 bins)')

plt.tight_layout()
features = list(ivdf.loc[ivdf['IV']>3,:]['Var'])

print(features)

train_df['pred_lgb_ivgt3'],test_df['pred_lgb_ivgt3']=lgbmodelling(features,train_df,test_df)





features = list(ivdf.loc[ivdf['IV']<=3,:]['Var'])

print(features)

train_df['pred_lgb_ivle3'],test_df['pred_lgb_ivle3']=lgbmodelling(features,train_df,test_df)





target = train_df['target']

features=['pred_lgb_ivgt3','pred_lgb_ivle3']

print(features)



train_df['pred_lgb_iv3'],test_df['pred_lgb_iv3']=lgbmodelling(features,train_df,test_df)
sub_df = pd.DataFrame({"ID_code":test_df["ID_code"].values})

sub_df["target"] = test_df['pred_lgb_iv3']

sub_df.to_csv("lgbm pred_lgb_iv3.csv", index=False)