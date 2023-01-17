# -*- coding: utf-8 -*-

# Title          : Automatic Model Selector
# Author         : Eunjong Kim
# Origin_Date    : 05/11/2018
# Revision_Date  : 05/14/2018
# Version        : '0.1.2'

import h2o
from h2o.automl import H2OAutoML

h2o.init()
# load_file Definition Example
df = h2o.import_file(path="../input/creditcard.csv", destination_frame="df")
print(df.head())
# Input parameters that are going to train (Target)
response_column = 'Class'
print(response_column)
# Output parameter train against input parameters
training_columns = df.columns.remove(response_column)
print(training_columns)
# Split data into train and testing
train, test = df.split_frame(ratios=[0.7])

# For regression     --- default
# For classification --- binary
response_type = 'binary'

# For binary classification, response should be a factor
if response_type == 'binary':
    train[response_column] = train[response_column].asfactor()
    test[response_column] = test[response_column].asfactor()
### AutoML

# Time to run the experiment
run_automl_for_seconds = 1000

# RUN AutoML
aml = H2OAutoML(max_runtime_secs=run_automl_for_seconds)

aml.train(x=training_columns, y=response_column,
          training_frame=train,
          leaderboard_frame=test)
# View the AutoML Leaderboard
lb = aml.leaderboard
print(lb)
# The leader model is stored here
aml.leader
print(aml.leader)
# predict
pred = aml.predict(test_data=test)
print(pred)
