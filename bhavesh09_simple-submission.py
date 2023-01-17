import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

test['scalar_coupling_constant'] = test['type'].map(train.groupby('type')['scalar_coupling_constant'].mean())

test[['id','scalar_coupling_constant']].to_csv('simple_submission.csv', index=False)