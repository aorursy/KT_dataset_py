import lightgbm as lgb

import pandas as pd

import numpy as np



from datetime import datetime

from pathlib import Path

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import RepeatedStratifiedKFold
path=Path("../input/")

train=pd.read_csv(path/"train.csv").drop("ID_code",axis=1)

test=pd.read_csv(path/"test.csv").drop("ID_code",axis=1)
## Inspiration from

#https://www.kaggle.com/jiweiliu/lgb-2-leaves-augment

def augment(train,num_n=1,num_p=2):

    newtrain=[train]

    

    n=train[train.target==0]

    for i in range(num_n):

        newtrain.append(n.apply(lambda x:x.values.take(np.random.permutation(len(n)))))

    

    for i in range(num_p):

        p=train[train.target>0]

        newtrain.append(p.apply(lambda x:x.values.take(np.random.permutation(len(p)))))

    return pd.concat(newtrain)

#df=oversample(train,2,1)
param = {

   "objective" : "binary",

    "metric" : "auc",

    "boosting": 'gbdt',

    "max_depth" : -1,

    "num_leaves" : 13,

    "learning_rate" : 0.01,

    "bagging_freq": 5,

    "bagging_fraction" : 0.4,

    "feature_fraction" : 0.05,

    "min_data_in_leaf": 80,

    "min_sum_heassian_in_leaf": 10,

    "tree_learner": "serial",

    "boost_from_average": "false",

    "bagging_seed" : 10,

    "verbosity" : 1,

}
result=np.zeros(test.shape[0])



rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5,random_state=10)

for counter,(train_index, valid_index) in enumerate(rskf.split(train, train.target),1):

    print (counter)

    

    #Train data

    t=train.iloc[train_index]

    t=augment(t)

    trn_data = lgb.Dataset(t.drop("target",axis=1), label=t.target)

    

    #Validation data

    v=train.iloc[valid_index]

    val_data = lgb.Dataset(v.drop("target",axis=1), label=v.target)

    

    #Training

    model = lgb.train(param, trn_data, 1000000, valid_sets = [val_data], verbose_eval=500, early_stopping_rounds = 3000)

    result +=model.predict(test)

submission = pd.read_csv(path/'sample_submission.csv')

submission['target'] = result/counter

filename="{:%Y-%m-%d_%H_%M}_sub.csv".format(datetime.now())

submission.to_csv(filename, index=False)