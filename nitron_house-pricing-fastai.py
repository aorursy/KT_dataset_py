# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from fastai import *

from fastai.tabular import *



import os

# Any results you write to the current directory are saved as output.
root=Path("../input")

train_path=root/'train.csv'

test_path=root/'test.csv'
test_df=pd.read_csv(test_path).fillna(0)

test_df.head()
train_df=pd.read_csv(train_path)

train_df.head()
train_df1=train_df.loc[0:1200]

train_df2=train_df.loc[1200:]
procs=[FillMissing,Categorify]
valid_idx=list(range(500,700))
dep_var='SalePrice'



# TODO Feature selection

cat_names=list(train_df.select_dtypes('object').columns.values)

cont_names=list(train_df.select_dtypes(['int64','float64']))

cont_names.remove('Id')

cont_names.remove('SalePrice')
test_db = TabularList.from_df(test_df,path=test_path,cont_names=cont_names,cat_names=cat_names)
data=(TabularList.from_df(train_df,path=train_path,cont_names=cont_names,cat_names=cat_names,procs=procs)

#       .split_by_idx(valid_idx)

      .split_by_rand_pct(0.1)

      .label_from_df(cols=dep_var,label_cls=FloatList,log=False)

      .add_test(test_db)

      .databunch())
data.show_batch(10)
learn=tabular_learner(data,layers=[4000,2000], metrics=[mean_absolute_error],path='.')
learn.fit_one_cycle(5,.01)
learn.save('s1')
learn.load('s1');
learn.unfreeze()
learn.lr_find(1e-10,1e+10)
learn.recorder.plot()
learn.fit_one_cycle(20,10)
vdf=train_df2
vdb=(TabularList.from_df(vdf,path=train_path,cont_names=cont_names,cat_names=cat_names,procs=procs)

     .split_by_rand_pct(.01)

     .label_from_df(cols=dep_var,label_cls=FloatList,log=False)

     .databunch())
learn.validate(vdb.valid_dl)
learn.validate(data.valid_dl)
i=2
data.valid_ds[i]
train_df1.iloc[500+i]["GarageArea"]
learn.predict(train_df1.iloc[500+i])[0].obj[0]
j=10

train_df2.iloc[j]['SalePrice']
learn.predict(train_df2.iloc[j])
sub_df = pd.DataFrame(columns=['Id','SalePrice']).astype({'Id':int,'SalePrice':float})
for index, rw in test_df.iterrows():

    price=learn.predict(rw)[0].obj[0]

    sub_df.loc[index]=[int(rw.Id),price]
sub_df=sub_df.astype({'Id':int,'SalePrice':float})

sub_df.head()
sub_df.to_csv('submission.csv',index=False)