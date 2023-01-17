# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#data = pd.read_csv('/kaggle/input/diploma-data-to-boost/data_for_forest_trained_200_clusters.csv', header = None)

# y = data[800].astype(int)

# data.drop(columns=[800], inplace = True)

# X = data.copy()
import h2o

from h2o.automl import H2OAutoML

h2o.init()

df = h2o.import_file('/kaggle/input/diploma-data-to-boost/data_for_forest_trained_200_clusters.csv', destination_frame="df")
train, valid, test = df.split_frame(ratios=[0.64,0.16], seed=123)

response = "C801"

train[response] = train[response].asfactor()

valid[response] = valid[response].asfactor()

test[response] = test[response].asfactor()

print("Number of rows in train, valid and test set : ", train.shape[0], valid.shape[0], test.shape[0])
predictors = df.columns[:-1]

aml = H2OAutoML(max_models = 10, seed = 124, max_runtime_secs=10000)

aml.train(x=predictors, y=response, training_frame=train, validation_frame=valid)
lb = aml.leaderboard

lb
preds = aml.leader.predict(test)
from sklearn.metrics import log_loss, roc_auc_score

print(log_loss(test[response].as_data_frame().values, preds['p1'].as_data_frame().values))

print(roc_auc_score(test[response].as_data_frame().values, preds['p1'].as_data_frame().values))