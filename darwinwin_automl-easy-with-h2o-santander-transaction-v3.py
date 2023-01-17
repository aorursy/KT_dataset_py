import h2o

from h2o.automl import H2OAutoML

import numpy as np  # linear algebra

import pandas as pd  #

from sklearn.metrics import roc_auc_score

import os

print(os.listdir("../input"))



h2o.init(max_mem_size='16G')
htrain = h2o.import_file('../input/train.csv')

htest_sub = h2o.import_file('../input/test.csv')



htrain = htrain.drop(['ID_code'])

htest_sub = htest_sub.drop(['ID_code'])

                            

x = htrain.columns

y = "target"

x.remove(y)

# For binary classification, response should be a factor

htrain[y] = htrain[y].asfactor()



print("Train set size:", htrain.shape, "Test set size:", htest_sub.shape)
htrain.head()
# FIT

aml = H2OAutoML(seed = 42, 

                # max_models=2, # for fast test

                max_runtime_secs = 31000,

                stopping_metric = "AUC") 

aml.train(x=x, y=y, training_frame=htrain,)
# View the AutoML Leaderboard

lb = aml.leaderboard

lb.head(rows=lb.nrows)  # Print all rows instead of default (10 rows)
# The leader model is stored here

aml.leader
# SUBMIT

sub_pred = aml.leader.predict(htest_sub)

sub_pred = sub_pred.as_data_frame()

print('predict shape:', sub_pred.shape)



sample_submission = pd.read_csv('../input/sample_submission.csv')

sample_submission['target'] = sub_pred.p1

sample_submission.to_csv('h2o_AutoML_submission_v3.csv', index=False)