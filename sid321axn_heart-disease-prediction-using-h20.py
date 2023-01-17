import h2o

import time

import seaborn

import itertools

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from h2o.estimators.glm import H2OGeneralizedLinearEstimator

from h2o.estimators.gbm import H2OGradientBoostingEstimator

from h2o.estimators.random_forest import H2ORandomForestEstimator



%matplotlib inline

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
h2o.init()
heart_df = h2o.import_file("../input/heart.csv", destination_frame="heart_df")
heart_df.head()
heart_df.describe()
for col in heart_df.columns:

    heart_df[col].hist()
plt.figure(figsize=(10,10))

corr = heart_df.cor().as_data_frame()

corr.index = heart_df.columns

sns.heatmap(corr, annot = True, cmap='RdYlGn', vmin=-1, vmax=1)

plt.title("Correlation Heatmap", fontsize=16)

plt.show()
train, valid, test = heart_df.split_frame(ratios=[0.6,0.1], seed=1234)

response = "target"

train[response] = train[response].asfactor()

valid[response] = valid[response].asfactor()

test[response] = test[response].asfactor()

print("Number of rows in train, valid and test set : ", train.shape[0], valid.shape[0], test.shape[0])
predictors = heart_df.columns[:-1]

gbm = H2OGradientBoostingEstimator()

gbm.train(x=predictors, y=response, training_frame=train)
print(gbm)
perf = gbm.model_performance(valid)

print(perf)
gbm_tune = H2OGradientBoostingEstimator(

    ntrees = 1000,

    learn_rate = 0.01,

    stopping_rounds = 20,

    stopping_metric = "AUC",

    col_sample_rate = 0.7,

    sample_rate = 0.7,

    seed = 1234

)      

gbm_tune.train(x=predictors, y=response, training_frame=train, validation_frame=valid)
gbm_tune.model_performance(valid).auc()
from h2o.grid.grid_search import H2OGridSearch



gbm_grid = H2OGradientBoostingEstimator(

    ntrees = 1000,

    learn_rate = 0.01,

    stopping_rounds = 20,

    stopping_metric = "AUC",

    col_sample_rate = 0.7,

    sample_rate = 0.7,

    seed = 1234

) 



hyper_params = {'max_depth':[4,6,8,10,12]}

grid = H2OGridSearch(gbm_grid, hyper_params,

                         grid_id='depth_grid',

                         search_criteria={'strategy': "Cartesian"})

#Train grid search

grid.train(x=predictors, 

           y=response,

           training_frame=train,

           validation_frame=valid)
print(grid)
sorted_grid = grid.get_grid(sort_by='auc',decreasing=True)

print(sorted_grid)
cv_gbm = H2OGradientBoostingEstimator(

    ntrees = 3000,

    learn_rate = 0.05,

    stopping_rounds = 20,

    stopping_metric = "AUC",

    nfolds=4, 

    seed=2018)

cv_gbm.train(x = predictors, y = response, training_frame = train, validation_frame=valid)

cv_summary = cv_gbm.cross_validation_metrics_summary().as_data_frame()

cv_summary
cv_gbm.model_performance(valid).auc()
from h2o.estimators import H2OXGBoostEstimator



cv_xgb = H2OXGBoostEstimator(

    ntrees = 1000,

    learn_rate = 0.05,

    stopping_rounds = 20,

    stopping_metric = "AUC",

    nfolds=4, 

    seed=2018)

cv_xgb.train(x = predictors, y = response, training_frame = train, validation_frame=valid)

cv_xgb.model_performance(valid).auc()
cv_xgb.varimp_plot()
from h2o.automl import H2OAutoML



aml = H2OAutoML(max_models = 10, max_runtime_secs=100, seed = 1)

aml.train(x=predictors, y=response, training_frame=train, validation_frame=valid)
lb = aml.leaderboard

lb