# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from fastai.tabular import *

from pathlib import Path

from tqdm import tqdm

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import roc_auc_score

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
path = Path('')



train = pd.read_csv('../input/train.csv')

train.drop(columns=['ID_code'],inplace=True)



tester =  pd.read_csv('../input/test.csv')

tester.drop(columns=['ID_code'],inplace=True)



#test = TabularList.from_df(tester.copy(), cat_names=cat_names, cont_names=cont_names)



dep_var = 'target'

cat_names = [] 

cont_names = train.columns

procs = [Normalize]



#valid_idx = range(len(train)-2000, len(train))
a = tester.columns

t = pd.Index(['target'])

a = a.append(t)



test_probs = np.zeros(200000)

nFolds = 12

BATCH_SIZE = (4096*2)
folds = StratifiedKFold(n_splits=nFolds, shuffle=False, random_state=99999)

for fold_, (trn_idx, val_idx) in tqdm(enumerate(folds.split(train.values, train['target'].values))):

    valid_idx = val_idx

    data = TabularDataBunch.from_df(path = path,df = train[a], dep_var = dep_var, valid_idx=valid_idx, procs=procs, cat_names=cat_names,bs=BATCH_SIZE, test_df=tester)

    learn = tabular_learner(data, layers=[100,50,25], metrics=accuracy)

    learn.fit_one_cycle(3, 1e-2)

    test_predicts, _ = learn.get_preds(ds_type=DatasetType.Test)

    test_probs += to_np(test_predicts[:, 1])
test_probs = test_probs/nFolds
# (cat_x,cont_x),y = next(iter(data.train_dl))

# for o in (cat_x, cont_x, y): print(to_np(o[:5]))

#learn.save('ch-2')

# learn.lr_find()

# learn.recorder.plot()

# preds = []

# for i in tqdm(range(0,len(tester))):

#     a = learn.predict(tester.iloc[i])

#     b = a[2].numpy()

#     preds.append(b[1])
sub = pd.read_csv('../input/sample_submission.csv')



sub['target'] = test_probs



sub.to_csv('submission.csv',index=False)