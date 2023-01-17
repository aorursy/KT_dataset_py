from fastai.tabular import *

import numpy as np

import pandas as pd

from pathlib import Path



path = Path('../input')
train_df = pd.read_csv(path/'train.csv')

train_df.pop('ID_code')

train_df_len = len(train_df)



test_df = pd.read_csv(path/'test.csv')

test_ID_codes_df = test_df.pop('ID_code')
valid_idx = range(round(0.99*train_df_len), train_df_len)

data: DataBunch = TabularDataBunch.from_df(path, train_df, dep_var='target', valid_idx=valid_idx, test_df=test_df)

learn = tabular_learner(data, layers=[200,100], metrics=accuracy, path='.')
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(1, 5e-2)
preds = learn.get_preds(ds_type=DatasetType.Test)

target_preds = preds[0][:,0]
test_df['ID_code'] = test_ID_codes_df

test_df['target'] = target_preds
test_df.to_csv('submission.csv', columns=['ID_code', 'target'], index=False)