import os

os.listdir('../input')
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline
import h2o
h2o.init()
dFrame = h2o.import_file('../input/heart-disease-uci/heart.csv')
dFrame.head()
dFrame.columns
dFrame.summary()
dFrame.describe()
dFrame.cor()
Corr = dFrame.cor().as_data_frame()

Corr.index = dFrame.columns

mask = np.triu(np.ones_like(Corr, dtype=np.bool))

f, ax = plt.subplots(figsize=(11, 9))

sns.heatmap(Corr, mask=mask, cmap='RdYlGn', vmax=.3, center=0,square=True, linewidths=.5, cbar_kws={"shrink": .5},annot=True)

 
train,valid, test = dFrame.split_frame(ratios=[0.6,0.1],seed=1234)



train["target"] = train["target"].asfactor()

valid["target"] = valid["target"].asfactor()

test["target"]  = test["target"].asfactor()
print(train.shape)

print("*"*20)

print(valid.shape)

print("*"*20)

print(test.shape)

print("*"*20)
predct = dFrame.columns[:-1]
from h2o.estimators.glm import H2OGeneralizedLinearEstimator

from h2o.estimators.gbm import H2OGradientBoostingEstimator

from h2o.estimators.random_forest import H2ORandomForestEstimator
gbm = H2OGradientBoostingEstimator()

gbm.train(x=predct,y="target",training_frame=train)
print(gbm)
prfm = gbm.model_performance(valid)

print(prfm)
tuning = H2OGradientBoostingEstimator(

    ntrees = 1000,

    learn_rate = 0.01,

    stopping_rounds = 22,

    stopping_metric = "AUC",

    col_sample_rate = 0.7,

    sample_rate = 0.8,

    seed = 1234

) 
tuning.train(x=predct, y="target", training_frame=train, validation_frame=valid)
print (tuning.model_performance(valid).auc()*100)
from h2o.estimators import H2OXGBoostEstimator
xgb = H2OXGBoostEstimator(ntrees=1000,learn_rate=0.05,stopping_rounds=20,stopping_metric="AUC",nfolds=10,seed=1234)
xgb.train(x=predct,y="target",training_frame=train, validation_frame=valid)
print(xgb.model_performance(valid).auc()*100)
xgb.varimp
xgb.varimp_plot(num_of_features =5)
#Use with autoML

from h2o.automl import H2OAutoML
autoML = H2OAutoML(max_models=15,max_runtime_secs=150,seed=3)

autoML.train(x=predct,y="target",training_frame = train, validation_frame=valid)
print("*"*102)

print(autoML.leaderboard)

print("*"*102)
#We can get the information as GBM is perfoming well with aprox ~90% area under curve and 0.39 log loss.
#Please upvote the kernal if this feels informative. :)