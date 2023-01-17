import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
test.insert(1,'target',train.target.mean())
x = pd.concat([train[['id','target']],test[['id','target']]])

x = x.sort_values(by='id').reset_index(drop=True)
a = x.rolling(5,min_periods=1).target.mean()
sub = pd.read_csv('../input/sample_submission.csv')

sub.target = a[~x.id.isin(train.id)].ravel()

sub.to_csv('leaksoup.csv',index=False)