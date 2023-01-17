import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
train_raw = pd.read_csv('../input/train.csv', header=0, index_col='ID')
train_raw.info()
train_raw['target'].value_counts()
string_cols = []

for col in train_raw.columns:
    if train_raw[col].dtype == 'object':
        string_cols.append(col)
string_cols
for sc in string_cols:
    print('{:4s} {:05d}'.format(sc, train_raw[sc].value_counts().values.size))
sorted(train_raw['v22'].value_counts().index.tolist())
