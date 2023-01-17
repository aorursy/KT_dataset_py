import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pandas import DataFrame



pd.set_option('display.max_rows', 100)
from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')

df_ss = pd.read_csv('../input/sample_submission.csv')
Y_train = df_train.SalePrice

df_train = df_train.drop(['Id', 'SalePrice'], axis=1)



dtypes = df_train.dtypes

cnt = df_train.count()

fields = DataFrame({'cnt': cnt, 'dtype': dtypes})



# drop sparse fields

cnt = df_train.count()

sparse_cols = fields[fields.cnt < 500].index

df_train = df_train.drop(sparse_cols, axis=1)

fields = fields.drop(sparse_cols)
# obj

fields_obj = fields.loc[fields.dtype=='object']

df_train_obj = df_train[fields_obj.index]



fields_obj['nunique'] = df_train_obj.apply(lambda s: s.nunique())
# num

fields_num = fields[fields.dtype != 'object']

df_train_num = df_train[fields_num.index]

fields_num = fields_num.join(df_train_num.describe().T)
fields_num.head()