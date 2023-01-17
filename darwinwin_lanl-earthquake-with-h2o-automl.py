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
print(os.listdir("../input/andrews-features-only"))

import h2o

from h2o.automl import H2OAutoML



h2o.init(max_mem_size='16G')
train = h2o.import_file("../input/andrews-features-only/X_tr.csv")

test = h2o.import_file("../input/andrews-features-only/X_test.csv")

y_train = h2o.import_file("../input/andrews-features-only/y_tr.csv")
train.head()
train['time_to_failure'] = y_train['time_to_failure']
train.shape
test.shape
x = test.columns

y = 'time_to_failure'
# Run AutoML for 20 base models (limited to 1 hour max runtime by default)

aml = H2OAutoML(max_models=50000, seed=55, max_runtime_secs=31000,nfolds=0,sort_metric="MAE",stopping_metric="MAE")

aml.train(x=x, y=y, training_frame=train)
# View the AutoML Leaderboard

lb = aml.leaderboard

lb.head(rows=lb.nrows)  # Print all rows instead of default (10 rows)
# The leader model is stored here

aml.leader
# If you need to generate predictions on a test set, you can make

# predictions directly on the `"H2OAutoML"` object, or on the leader

# model object directly



preds = aml.predict(test)
sample_submission = pd.read_csv('../input/LANL-Earthquake-Prediction/sample_submission.csv')

sample_submission['time_to_failure'] = preds.as_data_frame().values.flatten()

sample_submission.to_csv('h2o_submission_1.csv', index=False)