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
import h2o

print(h2o.__version__)

from h2o.automl import H2OAutoML



h2o.init(max_mem_size='16G')
%%time

train = h2o.import_file("../input/multi-cat-encodings/X_train_te.csv")

test = h2o.import_file("../input/multi-cat-encodings/X_test_te.csv")
train.head()

x = test.columns

y = 'target'

train[y] = train[y].asfactor()
aml = H2OAutoML(seed=47, max_runtime_secs=30000)

aml.train(x=x, y=y, training_frame=train, fold_column='fold_column')
# View the AutoML Leaderboard

lb = aml.leaderboard

lb.head(rows=lb.nrows)  # Print all rows instead of default (10 rows)
# The leader model is stored here

aml.leader
preds = aml.predict(test)

preds['p1'].as_data_frame().values.flatten().shape
sample_submission = pd.read_csv('../input/cat-in-the-dat-ii/sample_submission.csv')

sample_submission.shape
sample_submission['target'] = preds['p1'].as_data_frame().values

sample_submission.to_csv('h2o_automl_submission_4.csv', index=False)