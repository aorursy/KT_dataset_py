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
print(os.listdir("../input/fathers-day-specials-just-the-features"))
print(os.listdir("../input/mitsuru-features-only-train-and-test"))
import h2o

from h2o.automl import H2OAutoML



h2o.init(max_mem_size='16G')
train = h2o.import_file("../input/mitsuru-features-only-train-and-test/train_df_noindex.csv")

test = h2o.import_file("../input/mitsuru-features-only-train-and-test/test_df_noindex.csv")
train.head()
train.shape
test.head()
test.shape
x = test.columns

y = 'target'
# Run AutoML for 20 base models (limited to 1 hour max runtime by default)

aml = H2OAutoML(max_models=180, seed=47, max_runtime_secs=31000)

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
preds.as_data_frame().values.flatten()
sample_submission = pd.read_csv('../input/elo-merchant-category-recommendation/sample_submission.csv')

sample_submission['target'] = preds.as_data_frame().values.flatten()

sample_submission.to_csv('h2o_submission_1.csv', index=False)