# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os
import h2o

print(h2o.__version__)

from h2o.automl import H2OAutoML



h2o.init(max_mem_size='16G')
train = pd.read_csv('/kaggle/input/predict-volcanic-eruptions-ingv-oe/train.csv')

test = pd.read_csv('/kaggle/input/predict-volcanic-eruptions-ingv-oe/sample_submission.csv')
scaled_feature_df = pd.read_pickle('../input/ingv-volcanic-eruption-prediction-add-resampling/scaled_feature_df.pickle')

scaled_test_df = pd.read_pickle('../input/ingv-volcanic-eruption-prediction-add-resampling/scaled_test_df.pickle')
scaled_feature_df = scaled_feature_df.loc[:, ~scaled_feature_df.columns.str.startswith('sensor_4_')]

scaled_test_df = scaled_test_df.loc[:, ~scaled_test_df.columns.str.startswith('sensor_4_')]
train_h2o = h2o.H2OFrame(scaled_feature_df)

train_label_h2o = h2o.H2OFrame(train[['time_to_eruption']])

train_h2o['time_to_eruption'] = train_label_h2o['time_to_eruption']



test_feature_h2o = h2o.H2OFrame(scaled_test_df)
print(train_h2o.shape)

print(test_feature_h2o.shape)
x = test_feature_h2o.columns

y = 'time_to_eruption'
aml = H2OAutoML(max_models=1000, seed=121, include_algos = ["GBM"], # Let's just focus GBM model now

                max_runtime_secs=120*60) # set 120 minutes

aml.train(x=x, y=y, training_frame=train_h2o)
# View the AutoML Leaderboard

lb = aml.leaderboard

lb.head(rows=lb.nrows)  # Print all rows instead of default (10 rows)
# The leader model is stored here

aml.leader
# If you need to generate predictions on a test set, you can make

# predictions directly on the `"H2OAutoML"` object, or on the leader

# model object directly



preds = aml.predict(test_feature_h2o)
submission = pd.DataFrame()

submission['segment_id'] = test['segment_id']

submission['time_to_eruption'] = preds.as_data_frame().values.flatten()

submission.loc[submission['time_to_eruption']<0, 'time_to_eruption'] = 0 #make sure all prediction values are larger than 0

submission.to_csv('submission_recent.csv', header=True, index=False)
submission