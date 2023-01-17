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
df=pd.read_csv("../input/data-ready-for-model/csvfile.csv")

df
df_=df[df.columns[~df.columns.isin(['default.payment.next.month'])]]
import h2o

h2o.init()
data=h2o.H2OFrame(df)

data.head()
data_=h2o.H2OFrame(df_)

data_.head()
data['default.payment.next.month']=data['default.payment.next.month'].asfactor()

data.head()
#Import the Estimators

from h2o.estimators.gbm import H2OGradientBoostingEstimator  # import gbm estimator



#Import h2o grid search 

import h2o.grid 

from h2o.grid.grid_search import H2OGridSearch



model = H2OGradientBoostingEstimator(## more trees is better if the learning rate is small enough 

  ## here, use "more than enough" trees - we have early stopping

  ntrees = 10000,                                                            



  ## smaller learning rate is better (this is a good value for most datasets, but see below for annealing)

  learn_rate = 0.01,                                                         



  ## early stopping once the validation AUC doesn't improve by at least 0.01% for 5 consecutive scoring events

  stopping_rounds = 5, stopping_tolerance = 1e-4, stopping_metric = "AUC", 



  ## sample 80% of rows per tree

  sample_rate = 0.8,                                                       



  ## sample 80% of columns per split

  col_sample_rate = 0.8,                                                   



  ## fix a random number generator seed for reproducibility

  seed = 1234,                                                             



  ## score every 10 trees to make early stopping reproducible (it depends on the scoring interval)

  score_tree_interval = 10, nfolds=5, max_depth=3)   ## Instantiating the class





train, valid, test = data.split_frame([0.7, 0.15], seed=42)

gbm=model.train(x=data_.names,y='default.payment.next.month', training_frame=train, model_id="GBM",validation_frame=valid)
model.cross_validation_metrics_summary()
# plot decision tree

from numpy import loadtxt

from xgboost import XGBClassifier

from xgboost import plot_tree

import matplotlib.pyplot as plt

# load data

df=pd.read_csv("../input/data-ready-for-model/csvfile.csv")

# split data into X and y

X = df[data_.names]

y = df['default.payment.next.month']

# fit model no training data

model = XGBClassifier()

model.fit(X, y)

# plot single tree

plot_tree(model)

plt.show()
data['default.payment.next.month'] = data['default.payment.next.month'].asfactor()    



train, test = data.split_frame([0.7], seed=42)



y = 'default.payment.next.month'



x = data_.names



gbm= H2OGradientBoostingEstimator(distribution="AUTO")

%time gbm.train(x=x, y=y, training_frame=train)
perf=gbm.model_performance(test)
perf.auc()
gbm.varimp_plot(20)