# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import h2o

import matplotlib as plt

%matplotlib inline

from h2o.automl import H2OAutoML
h2o.init()
loan_level = h2o.import_file("https://s3.amazonaws.com/data.h2o.ai/H2O-3-Tutorials/loan_level_50k.csv")
loan_level.head()
loan_level['DELINQUENT'].table()
loan_level['ORIGINAL_INTEREST_RATE'].hist()
train, test = loan_level.split_frame([0.8],seed=42)
print('train: %d test: %d' % (train.nrows, test.nrows))
y = 'DELINQUENT'

ignore = ['DELINQUENT', 'PREPAID', 'PREPAYMENT_PENALTY_MORTGAGE_FLAG', 'PRODUCT_TYPE']

x = list(set(train.names)-set(ignore))
H2OAutoML(nfolds=5, max_runtime_secs=3600, max_models=None, stopping_metric='AUTO',

         stopping_tolerance=None, stopping_rounds=3, seed=None, project_name=None)
aml = H2OAutoML(max_models=25, max_runtime_secs_per_model=30, seed=42, project_name='classification',

                balance_classes=True, class_sampling_factors=[0.5,1.25])

%time aml.train(x=x, y=y, training_frame=train)
lb = aml.leaderboard

lb.head(rows=lb.nrows)
from h2o.automl import get_leaderboard

lb2 = get_leaderboard(aml, extra_columns='ALL')

lb.head(rows=lb2.nrows)