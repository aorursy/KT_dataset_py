# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from fastai.tabular import *



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



path = "/kaggle/input/titanic"

train_path = "/kaggle/input/titanic/train.csv"

train_data = pd.read_csv(train_path,  usecols=["PassengerId", "Survived", "Sex"])

# Any results you write to the current directory are saved as output.



test_path = "/kaggle/input/titanic/test.csv"

test_data = pd.read_csv(test_path, usecols=["PassengerId", "Sex"])





catagory_names = ["PassengerId", "Sex"]

#continues_names = ['Age', 'Fare',]

dependet_var = "Survived"

procs = [Categorify]

# procs = [FillMissing, Categorify, Normalize]



# test = (TabularList.from_df(test_data , path=path, cat_names=catagory_names, procs=procs).split_none().label_const(dependet_var).databunch())

# test

test = TabularList.from_df(test_data , path=path, cat_names=catagory_names, procs=procs)

data = (TabularList.from_df(train_data, path=path, cat_names=catagory_names, procs=procs)

                           .split_by_idx(list(range(200,400)))

                           .label_from_df(cols=dependet_var)

                            .add_test(test)

                           .databunch())



# data = (TabularList.from_df(df, path=path, cat_names=cat_vars, cont_names=cont_vars, procs=procs,)

#                 .split_by_idx(valid_idx)

#                 .label_from_df(cols=dep_var, label_cls=FloatList, log=True)

#                 .add_test(TabularList.from_df(test_df, path=path, cat_names=cat_vars, cont_names=cont_vars))

#                 .databunch())



learn = tabular_learner(data, layers=[200,100], metrics=error_rate)

learn.model_dir = '/kaggle/working/'

# learn.lr_find()

# learn.recorder.plot()



learn.fit_one_cycle(7, 1e-03)



predictions=learn.get_preds(DatasetType.Test)[1].numpy()



final_df = pd.DataFrame({ 'PassengerId' : test_data['PassengerId'] , 'Survived': predictions })

final_df.to_csv('submission2.csv', header=True, index=False)
