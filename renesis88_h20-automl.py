import numpy as np
import pandas as pd
import h2o
from h2o.automl import H2OAutoML
h2o.init(max_mem_size='16G')
train = h2o.import_file("/home/Documents/Mnist/train.csv")
test = h2o.import_file("/home/Documents/Mnist/test.csv")
x = train.columns[1:]
y = 'label'
train[y] = train[y].asfactor()

aml = H2OAutoML(max_models=30, seed=45, max_runtime_secs=28800)
aml.train(x=x, y=y, training_frame=train)
lb = aml.leaderboard
lb.head(rows=lb.nrows)  # Print all rows instead of default (10 rows)

aml.leader # The leader model is stored here
preds = aml.predict(test)
preds['p1'].as_data_frame().values.shape
# preds
sample_submission = pd.read_csv('/home/Documents/Mnist/sample_submission.csv')
sample_submission.shape
sample_submission['Label'] = preds['predict'].as_data_frame().values
sample_submission.to_csv('h2o_automl_submission_1.csv', index=False)
sample_submission.head()