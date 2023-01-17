# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import *
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
# Any results you write to the current directory are saved as output.
from fastai.tabular import *
train_df = pd.read_csv('../input/elo-world-high-score-without-blending/train_df.gz', compression='gzip', header=0,
                      sep=',',quotechar='"')
test_df = pd.read_csv('../input/elo-world-high-score-without-blending/test_df.gz', compression='gzip', header=0,
                      sep=',',quotechar='"')
train_df.shape, test_df.shape
df_indep = train_df.drop('target',axis=1)
n_valid = 40000 #~20% of the training set
n_trn = len(train_df)-n_valid
n_trn, n_valid
cat_flds = [n for n in df_indep.columns if train_df[n].nunique()<50 and n != 'outliers']
','.join(cat_flds)
len(cat_flds)
for df in [train_df, df_indep, test_df]:
    df.drop('first_active_month', axis=1, inplace=True)
for df in [train_df, df_indep, test_df]:
    for n in cat_flds: 
        df[n] = df[n].astype('category').cat.as_ordered()
    df['card_id'] = df['card_id'].astype('category').cat.as_ordered()
    df['card_id_code'] = df.card_id.cat.codes
    df.drop('card_id', axis=1, inplace=True)
train_df.shape, test_df.shape
cont_flds = [n for n in df_indep.columns if n not in cat_flds and n!= 'outliers']
','.join(cont_flds)
procs=[FillMissing, Categorify, Normalize] #self-explanatory - neural nets like normalised values
len(cont_flds), len(cat_flds)
dep_var = 'target'
df = train_df[cat_flds + cont_flds + [dep_var]].copy()
df[dep_var].head()
path = Path('../input/') #we need to give some path - doesn't matter
data = (TabularList.from_df(df, path=path, cat_names=cat_flds, cont_names=cont_flds, procs=procs)
                   .split_by_idx(range(n_valid))
                   .label_from_df(cols=dep_var, label_cls=FloatList, log=False)
                   .databunch())
min_y = np.min(train_df['target'])*1.2
max_y = np.max(train_df['target'])*1.2
y_range = torch.tensor([min_y, max_y], device=defaults.device)
learn = tabular_learner(data, layers=[1000,500], ps=[0.001,0.01], emb_drop=0.04, 
                        y_range=y_range, metrics=rmse)
learn.model
len(data.train_ds.cont_names)
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(5, 1e-2, wd=0.2)
learn.recorder.plot_losses()
# learn.fit_one_cycle(5, 3e-4)
# learn.fit_one_cycle(4, 5e-2, wd=0.2)
# learn.fit_one_cycle(3, 1e-2, wd=0.2)
learn.predict(df.iloc[0])
