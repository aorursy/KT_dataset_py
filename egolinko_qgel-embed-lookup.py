import numpy as np 

import pandas as pd

from sklearn.model_selection import train_test_split, KFold, TimeSeriesSplit,RandomizedSearchCV

import lightgbm as lgb

# will require to pip install qGEL

import qGEL
train=pd.read_csv('../input/cat-in-the-dat/train.csv')

test=pd.read_csv('../input/cat-in-the-dat/test.csv')



my_vars=train.drop(['id', 'target'], axis=1).columns
def make_embed(col_name):

    my_samp=train[col_name].astype('str').to_frame().sample(5000)

    my_dummies=pd.get_dummies(my_samp[col_name])

    my_emb_, v_t, mb = qGEL.qgel(my_dummies, k=20)

    my_embed=pd.concat([my_samp[col_name].reset_index().drop('index', axis=1), 

                        pd.DataFrame(my_emb_)], 

                       axis=1, sort=True).drop_duplicates()

    my_embed.columns=[col_name]+[col_name+'_'+e for e in map(str, range(0, my_emb_.shape[1]))]

    return my_embed
emb_lkup=[make_embed(v) for v in my_vars]
l_tr=[]

for i in range(0,len(my_vars)):

    l_tr.append(pd.merge(train[my_vars].astype('str'),emb_lkup[i], on=my_vars[i], how='left'))

tr_emb=pd.concat(l_tr, axis=1).drop(my_vars, axis=1)

tr_emb.columns=["emb"+e for e in map(str,range(0, len(tr_emb.columns)))]



l_te=[]

for i in range(0,len(my_vars)):

    l_te.append(pd.merge(test[my_vars].astype('str'),emb_lkup[i], on=my_vars[i], how='left'))

te_emb=pd.concat(l_te, axis=1).drop(my_vars, axis=1)

te_emb.columns=["emb"+e for e in map(str,range(0, len(te_emb.columns)))]



tr_emb.shape, te_emb.shape
X_train,X_test,y_train,y_test=train_test_split(tr_emb, train['target'], test_size=0.0001)
# https://www.kaggle.com/a03102030/compare-logistic-lgbm

X_train=X_train.astype(float)

X_test=X_test.astype(float)

lgb_train = lgb.Dataset(X_train, y_train)  

lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train) 



params = {  

    'boosting_type': 'gbdt',  

    'objective': 'binary',  

    'learning_rate' : 0.01,

    'num_leaves' : 500, 

    'feature_fraction' : 0.75,

    'bagging_fraction' : 0.75,

    'metric': {'binary_logloss', 'auc'}

}  



gbm = lgb.train(params,  

                lgb_train,  

                num_boost_round=5000,  

                valid_sets=lgb_eval,  

                early_stopping_rounds=100) 



LGBM_TEST=gbm.predict(te_emb, num_iteration=gbm.best_iteration) 

pd.DataFrame({'id':test.id,'target':LGBM_TEST}).to_csv('submission.csv', index=False)