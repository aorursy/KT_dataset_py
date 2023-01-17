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
path = Path('../input')
!ls {path}
df = pd.read_csv(f'{path}/train.csv', index_col='ID_code')
df.head()
df.describe()
# Describe the df where the target == 1 

df[df['target']==1].describe()
len(df)
# describe the df where target == 0

df[df['target']==0].describe()
df_test = pd.read_csv(f'{path}/test.csv', index_col='ID_code')
df_test.describe()
df['var_0'].unique()
for n in df.columns:

    print(n, ':', len(df[n].unique()))
dep_var = 'target'
cont_list, cat_list = cont_cat_split(df=df, max_card=20, dep_var=dep_var)
procs = [FillMissing, Categorify, Normalize]
test = TabularList.from_df(df_test, cat_names=cat_list, cont_names=cont_list, procs=procs)
data = (TabularList.from_df(df, path=path, cont_names=cont_list, cat_names=cat_list, procs=procs)

        .split_by_rand_pct(0.1)

        .label_from_df(dep_var)

        .add_test(test, label=0)

        .databunch())
data.batch_size = 128
learn = tabular_learner(data, layers=[ 100 , 100], metrics=accuracy, path=('.'), wd=1e-2)
device = "cuda" if torch.cuda.is_available() else "cpu"

print(device)
data.device = device
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(6, 5e-03)
predictions, *_ = learn.get_preds(DatasetType.Test)
predictions
labels = predictions[:,1]
#learn.fit_one_cycle(3, 1e-02)
#labels = np.argmax(predictions, 1)
#labels
#df_test.index
final_df = pd.DataFrame({'ID_code': df_test.index, 'target': labels})
final_df.to_csv('submission.csv', index=False)