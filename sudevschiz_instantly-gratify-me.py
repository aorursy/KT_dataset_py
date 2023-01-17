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
train_origin = pd.read_csv("../input/train.csv")
train_origin.head()
train_origin.shape
train_origin.target.sum()/train_origin.target.shape
import h2o

from h2o.automl import H2OAutoML
h2o.init()
train_hf = h2o.H2OFrame(train_origin)
train_hf['target'] = train_hf['target'].asfactor()
x = [elem for elem in train_hf.columns if elem not in ('id','target')]

y = 'target'
# Run AutoML for 20 base models (limited to 1 hour max runtime by default)

aml = H2OAutoML(max_models=20,

                seed=1,

                nfolds = 5

               )

aml.train(x=x, y=y, training_frame=train_hf)



# View the AutoML Leaderboard

lb = aml.leaderboard

lb.head(rows=lb.nrows)  # Print all rows instead of default (10 rows)

#Load test data

test_hf = h2o.import_file(path='../input/test.csv')
#Predict using the best model

preds = aml.predict(test_hf)
sub = pd.read_csv('../input/sample_submission.csv')

preds = preds.as_data_frame()

sub['target'] = preds['p1']

sub.to_csv('base_automl_submission.csv',index=False)