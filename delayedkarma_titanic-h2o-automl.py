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
import h2o
from h2o.automl import H2OAutoML

h2o.init(max_mem_size='4G')
%%time
train = h2o.import_file("../input/train.csv")
test = h2o.import_file("../input/test.csv")
train.describe()
x = train.columns
y = "Survived"
# For binary classification, response should be a factor
train[y] = train[y].asfactor()
x.remove(y)
%%time
# Run AutoML for 10 base models (limited to 1 hour max runtime by default)
aml = H2OAutoML(max_models=10, seed=42, max_runtime_secs=7200)
aml.train(x=x, y=y, training_frame=train)
lb = aml.leaderboard
lb.head(rows=lb.nrows)  # Print all rows instead of default (10 rows)
aml.leader # Best model
# If you need to generate predictions on a test set, you can make
# predictions directly on the `"H2OAutoML"` object, or on the leader
# model object directly

preds = aml.predict(test)
predictions = preds[0].as_data_frame().values.flatten()
sample_submission = pd.read_csv('../input/gender_submission.csv')
sample_submission['Survived'] = predictions
sample_submission.to_csv('h2O_titanic_1.csv', index=False)