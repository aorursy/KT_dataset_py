# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



import h2o

from h2o.automl import H2OAutoML

h2o.init()

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

df = h2o.H2OFrame(train)
df.head()
df['target'] = df['target'].asfactor()
y = "target"

x = df.columns[2:]
#max_runtime_secs= 3600, sort_metric='AUC'

aml = H2OAutoML(max_runtime_secs= 3600*6, max_models=60, sort_metric='AUC')

aml.train(x = x, y = y, training_frame = df)
lb = aml.leaderboard

lb.head(rows=lb.nrows) # Entire leaderboard
test = pd.read_csv('../input/test.csv')

hf = h2o.H2OFrame(test)
hf.head()
preds = aml.predict(hf)
preds = preds.as_data_frame()

preds['p_p0'] = np.exp(preds['p0'])

preds['p_p1'] = np.exp(preds['p1'])

preds['sm'] = preds['p_p1'] / (preds['p_p0'] + preds['p_p1'])
preds.head()
sub = pd.read_csv('../input/sample_submission.csv')

sub.target = preds['sm']

sub.to_csv('sub.csv', index=False)
h2o.save_model(aml.leader, path = ".")