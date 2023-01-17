import numpy as np

import pandas as pd

import h2o

from h2o.automl import H2OAutoML

h2o.init(max_mem_size='16G')
train_path = '../input/train.csv'

test_path = '../input/test.csv'
col_types = {'target': 'enum'}
train = h2o.import_file(path=train_path, col_types=col_types)
test = h2o.import_file(path=test_path)
y = 'target'

X = [name for name in train.columns if name not in ['ID_code', y]]
train[y] = train[y].asfactor()
model = H2OAutoML(max_models=50,

                  max_runtime_secs = 18000,

                seed=12345)

model.train(x=X, y=y, training_frame=train)
lb = model.leaderboard

lb.head(rows=lb.nrows)
model.leader
result = model.leader.predict(test)

result
sub = test.cbind(result[0])
sub = sub[['ID_code','predict']]

sub = sub.rename(columns={'predict':'target'})
sub = sub.as_data_frame()

sub.to_csv('submission.csv',index=False)
h2o.cluster().shutdown(prompt=True)