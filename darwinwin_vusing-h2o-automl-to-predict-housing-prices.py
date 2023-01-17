# You can easily install the library using pip

!pip install h2o
# And then load the libraries you'll use in this notebook

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline





import h2o

from h2o.automl import H2OAutoML
# Initialize your cluster

h2o.init()
train = h2o.import_file('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

test = h2o.import_file('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

train.head()
train = train[train['GrLivArea'] < 4500]

train['SalePrice'] = train['SalePrice'].log1p()
# Identify predictors and response

x = [col for col in train.columns if col not in ['Id','SalePrice']]

y = 'SalePrice'

test_id = test['Id']
aml = H2OAutoML(max_runtime_secs=90000, seed = 99, stopping_metric = 'RMSLE')

aml.train(x = x, y = y, training_frame = train)
lb = aml.leaderboard; lb
aml.leader
aml.leader.varimp_plot()
preds = aml.leader.predict(test)
# Convert results back(they had been transformed using log, remember?) and save them in a csv format.

result = preds.expm1()

sub = test_id.cbind(result)

sub.columns = ['Id','SalePrice']

sub = sub.as_data_frame()

sub.to_csv('submission.csv', index = False)
import pandas as pd

sample_submission = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")

test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")

train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")