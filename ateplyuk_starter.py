import pandas as pd

import numpy as np

import lightgbm as lgb
tr_df = pd.read_csv("../input/train.csv") 

ts_df = pd.read_csv("../input/test.csv")



tr_df.head()
tr = tr_df.drop(['id','target'], axis=1)

ts = ts_df.drop('id', axis=1)

t = tr_df["target"].values
prd = ts_df[['id']].copy()

prd['target'] = 0
pm = {'objective': 'binary'}
%%time

cn = 'wheezy-copper-turtle-magic'

for i in range(512):    

    x = (tr[cn]==i)

    j = (ts[cn]==i)    

    l_tr = lgb.Dataset(tr[x].values, label=t[x])    

    l_h = lgb.cv(pm,l_tr,metrics=['auc'])                     

    bst = np.array(l_h['auc-mean']).argmax()

    ml = lgb.train(pm, l_tr, num_boost_round=bst)

    prd.loc[j,'target'] = ml.predict(ts[j].values,num_iteration=ml.best_iteration)
prd.head()
prd.to_csv("submission.csv", index=False)