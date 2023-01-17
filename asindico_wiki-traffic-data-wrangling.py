import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
train = pd.read_csv('../input/train_1.csv')
train.head()
keys = pd.read_csv('../input/key_1.csv')


keys.head()
#train.merge(keys,on='Page',how='outer')



train.iloc[[2]].isnull().sum(axis=1)
train.iloc[train.iloc[[2]].notnull()]