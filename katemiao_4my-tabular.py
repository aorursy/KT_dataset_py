# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from fastai.tabular import *
path = untar_data(URLs.ADULT_SAMPLE)

df = pd.read_csv(path/'adult.csv')

df.head()
df.head()
dep_var = 'salary'

cat_names = ['workclass', 'education', 'marital-status', 'occupation', 'relationship','race']

cont_names = ['age','fnlwgt','education-num']

procs = [FillMissing, Categorify, Normalize]
test = TabularList.from_df(df.iloc[800:1000].copy(), path=path, cat_names=cat_names, cont_names=cont_names)
data = (TabularList.from_df(df, path=path, cat_names=cat_names, cont_names=cont_names, procs=procs)

       .split_by_idx(list(range(800,1000)))

       .label_from_df(cols=dep_var)

       .add_test(test)

       .databunch())
data.show_batch()
learn = tabular_learner(data, layers=[200,100], metrics=accuracy)
learn.fit(1, 1e-2)
row = df.iloc[0]
learn.predict(row)
df['salary'].value_counts()