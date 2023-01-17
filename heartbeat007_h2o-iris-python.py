import h2o
h2o.init()
import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
## import data 

!ls 
df = h2o.import_file('../input/irisdataset/Iris.csv')
df.head()


df = h2o.import_file('../input/irisdataset/Iris.csv',col_names=['id','SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm','Species'],

                    col_types=['numeric',"numeric", "numeric", "numeric", "numeric", "enum"])
df.head()
dfp = pd.read_csv('../input/irisdataset/Iris.csv')
df = h2o.H2OFrame(dfp)
df.head()
df.names
df.types
df.frame_id
df[['SepalLengthCm']].max()
df[['SepalLengthCm']].min()
df[['SepalLengthCm']].mean()
df[['SepalLengthCm']].hist()
df[['SepalWidthCm']].hist()
df[['PetalWidthCm']].hist()
df[['PetalLengthCm']].hist()
df.head()
## dropping column
df.drop('Id',axis=1)
###or
df=df[:,1:]   ## take the datafrmame but lose the first column
df.head()
## applying seeding

df.frame_id
## describe data
df.describe()
df['sepal_ratio'] = df['SepalLengthCm']/df['SepalWidthCm']
df.head()
df['petal_ratio'] = df['PetalLengthCm']/df['PetalWidthCm']
df.head()
df.cor() 
dfp.corr()
sns.heatmap(dfp.corr())
dfp.corr(method='spearman')
sns.heatmap(dfp.corr(method='spearman'))
## in H20

train_h,test_h,valid_h = df.split_frame([0.6,0.2])
train_h
test_h
valid_h
from sklearn.model_selection import train_test_split
xtrain,ytrain,xtest,ytest = train_test_split(dfp.drop('Species',axis=1),dfp[['Species']])
xtrain.head()
ytrain.head()
xtest.head()
ytest.head()
#h2o.exportFile(d, "/path/to/d.csv")

#h2o.exportFile(d, "s3://mybucket/d.csv")

#h2o.exportFile(d, "s3://<AWS_ACCESS_KEY>:<AWS_SECRET_KEY>@mybucket/d.csv")

#h2o.exportFile(d, "hdfs://namenode/path/to/d.csv")
h2o.export_file(df,'export.csv')
!ls
!cat export.csv
df.as_data_frame().plot()
sns.heatmap(df.as_data_frame().corr())
sns.heatmap(df.as_data_frame().corr(method='spearman'))
df.as_data_frame().plot.barh()
d = df.as_data_frame()
pd.plotting.scatter_matrix(d)
## you can import the whole csv in  a folder

train_h
y = 'Species'
x = df.names
x.remove(y)
## we will do a grid search for finding the best parameter
import math

from h2o.estimators.gbm import H2OGradientBoostingEstimator

from h2o.grid.grid_search import H2OGridSearch



#We only provide the required parameters, everything else is default

gbm = H2OGradientBoostingEstimator()

gbm.train(x=x, y=y, training_frame=train_h)



## Show a detailed model summary

print (gbm)



perf = gbm.model_performance(valid_h)

print(perf)
## with valid dataframne

cv_gbm = H2OGradientBoostingEstimator(nfolds = 4, seed = 0xDECAF)

cv_gbm.train(x = x, y = y, training_frame = train_h.rbind(valid_h))
cv_gbm
## now with grid search (this may take time depending on the spec of your computer .altough its run on jvm dont take too much time)
gbm_params1 = {'learn_rate': [0.01, 0.1],

                'max_depth': [3, 5, 9],

                'sample_rate': [0.8, 1.0],

                'col_sample_rate': [0.2, 0.5, 1.0]}
gbm_grid1 = H2OGridSearch(model=H2OGradientBoostingEstimator,

                          grid_id='gbm_grid1',

                          hyper_params=gbm_params1)
gbm_grid1.train(x=x, y=y,

                training_frame=train_h,

                validation_frame=valid_h,

                ntrees=100,stopping_metric = "AUC",

                seed=1)
gbm_grid1
## RANDOm GRID search 
gbm_params2 = {'learn_rate': [i * 0.01 for i in range(1, 11)],

                'max_depth': list(range(2, 11)),

                'sample_rate': [i * 0.1 for i in range(5, 11)],

                'col_sample_rate': [i * 0.1 for i in range(1, 11)]}



# Search criteria

search_criteria = {'strategy': 'RandomDiscrete', 'max_models': 36, 'seed': 1}



# Train and validate a random grid of GBMs

gbm_grid2 = H2OGridSearch(model=H2OGradientBoostingEstimator,

                          grid_id='gbm_grid2',

                          hyper_params=gbm_params2,

                          search_criteria=search_criteria)

gbm_grid2.train(x=x, y=y,

                training_frame=train_h,

                validation_frame=valid_h,

                ntrees=100,

                seed=1)





gbm_grid2
gbm_gridperf2 = gbm_grid2.get_grid(sort_by='mse', decreasing=True)
gbm_gridperf2



# Grab the top GBM model, chosen by validation AUC

best_gbm2 = gbm_gridperf2.models[0]



# Now let's evaluate the model performance on a test set

# so we get an honest estimate of top model performance

#best_gbm_perf2 = best_gbm2.model_performance(test)

best_gbm2


best_gbm_perf2 = best_gbm2.model_performance(test_h)



best_gbm_perf2.mse()
best_gbm_perf2.rmse()
## doing this thing again witha  differene
predict = best_gbm2.predict(test_h)
predict
# Predict the contributions using the GBM model and test data.

staged_predict_proba = best_gbm2.staged_predict_proba(test_h)
staged_predict_proba
conf = best_gbm2.confusion_matrix(test_h)
conf
from h2o.estimators.random_forest import H2ORandomForestEstimator

m = H2ORandomForestEstimator(

ntrees=100,

stopping_metric="misclassification",

stopping_rounds=3,

stopping_tolerance=0.02, #2%

max_runtime_secs=60,

model_id="RF:stop_test"

)

m.train(x, y, train_h, validation_frame=valid_h)
m
pref = m.model_performance(valid_h)
pref
cv_m = H2OGradientBoostingEstimator()

cv_m.train(x = x, y = y, training_frame = train_h.rbind(valid_h))
cv_m
import h2o.grid

g = h2o.grid.H2OGridSearch(

h2o.estimators.H2ORandomForestEstimator(

nfolds=10

),

hyper_params={

"ntrees": [50, 100, 120],

"max_depth": [40, 60],

"min_rows": [1, 2]

}

)

g.train(x, y, train_h)
g_gridperf2 = g.get_grid(sort_by='mse', decreasing=True)
g_gridperf2
g.confusion_matrix(test_h)
from h2o.estimators import naive_bayes
nv = naive_bayes.H2ONaiveBayesEstimator()
nv.train(x,y,train_h)
nv
pred = nv.predict(test_h)
pred
nv.confusion_matrix(test_h)
nv.mse()
nv.rmse()
nv.cross_validation_metrics_summary()
## this is the pojo model for the naive bias classifier
nv.download_pojo()
cv_m = naive_bayes.H2ONaiveBayesEstimator()

cv_m.train(x = x, y = y, training_frame = train_h.rbind(valid_h))
cv_m
from h2o.automl import H2OAutoML
aml = H2OAutoML(max_models=25, seed=1)

aml.train(x=x, y=y, training_frame=train_h)
lb = aml.leaderboard

lb
preds = aml.leader.predict(test_h)
preds
preds = aml.predict(test_h)
preds
aml.sort_metric
aml.leaderboard


# save the model

model_path = h2o.save_model(model=nv, path="/tmp/mymodel", force=True)



print (model_path)





# load the model

saved_model = h2o.load_model(model_path)
saved_model
from h2o.estimators.stackedensemble import H2OStackedEnsembleEstimator
nfolds = 5



# There are a few ways to assemble a list of models to stack together:

# 1. Train individual models and put them in a list

# 2. Train a grid of models

# 3. Train several grids of models

# Note: All base models must have the same cross-validation folds and

# the cross-validated predicted values must be kept.





# 1. Generate a 2-model ensemble (GBM + RF)



# Train and cross-validate a GBM

my_gbm = H2OGradientBoostingEstimator(

                                      ntrees=10,

                                      max_depth=3,

                                      min_rows=2,

                                      learn_rate=0.2,

                                      nfolds=nfolds,

                                      fold_assignment="Modulo",

                                      keep_cross_validation_predictions=True,

                                      seed=1)

my_gbm.train(x=x, y=y, training_frame=train_h)





# Train and cross-validate a RF

my_rf = H2ORandomForestEstimator(ntrees=50,

                                 nfolds=nfolds,

                                 fold_assignment="Modulo",

                                 keep_cross_validation_predictions=True,

                                 seed=1)

my_rf.train(x=x, y=y, training_frame=train_h)





# Train a stacked ensemble using the GBM and GLM above

ensemble = H2OStackedEnsembleEstimator(model_id="my_ensemble_binomiale",

                                       base_models=[my_gbm, my_rf])

ensemble.train(x=x, y=y, training_frame=train_h)



# Eval ensemble performance on the test data

perf_stack_test = ensemble.model_performance(test_h)
# Compare to base learner performance on the test set

perf_gbm_test = my_gbm.model_performance(test_h)

perf_rf_test = my_rf.model_performance(test_h)

baselearner_best_auc_test = max(perf_gbm_test.mse(), perf_rf_test.mse())

stack_auc_test = perf_stack_test.mse()

print("Best Base-learner Test MSE:  {0}".format(baselearner_best_auc_test))

print("Ensemble Test MSE:  {0}".format(stack_auc_test))



# Generate predictions on a test set (if neccessary)

pred = ensemble.predict(test_h)
pred
from h2o.estimators import deeplearning

m = h2o.estimators.deeplearning.H2ODeepLearningEstimator()

m.train(x, y, train_h)

p = m.predict(test_h)
p
r2 = m.r2()

mse = m.mse()

rmse = m.rmse()

r2
from h2o.estimators.deeplearning import H2ODeepLearningEstimator
model = H2ODeepLearningEstimator(

distribution="multinomial",

activation="RectifierWithDropout",

hidden=[128,128,128,128],

input_dropout_ratio=0.2,

sparse=True,

l1=1e-5,

epochs=10)
model.train(

x=x,

y=y,

training_frame=train_h,

validation_frame=test_h)
model
model.predict(test_h)
model_cv = H2ODeepLearningEstimator(

distribution="multinomial",

activation="RectifierWithDropout",

hidden=[32,32,32],

input_dropout_ratio=0.2,

sparse=True,

l1=1e-5,

epochs=100,

nfolds=5)
model_cv.train(

x=x,

y=y,

training_frame=train_h)
# View specified parameters of the Deep Learning model

model.params
model_cv
model_cv.model_performance(train=True)

model_cv.model_performance(valid=True)
model.mse(valid=True)



# Cross-validated MSE

model_cv.mse(xval=True)
model_cv.predict(test_h).head()
model_vi = H2ODeepLearningEstimator(

distribution="multinomial",

activation="RectifierWithDropout",

hidden=[32,32,32],

input_dropout_ratio=0.2,

sparse=True,

l1=1e-5,

epochs=10,

variable_importances=True)
model_vi.train(

x=x,

y=y,

training_frame=train_h,

validation_frame=test_h)
model_vi.varimp()
## grid Search in python deep learning book Deep Learning booket page 35
hidden_opt = [[32,32],[32,16,8],[100]]

#l1_opt = [1e-4,1e-3]

hyper_parameters = {"hidden":hidden_opt}

from h2o.grid.grid_search import H2OGridSearch

model_grid = H2OGridSearch(H2ODeepLearningEstimator,

hyper_params=hyper_parameters)

model_grid.train(x=x, y=y,

distribution="multinomial", epochs=1000,

training_frame=train_h, validation_frame=test_h,

score_interval=2, stopping_rounds=3,

stopping_tolerance=0.05,

stopping_metric="misclassification")
model_grid
model_grid.confusion_matrix(test_h)
model_grid.r2
model_grid.mse()
hidden_opt =[[17,32],[8,19],[32,16,8],[100],[10,10,10,10]]

l1_opt = [s/1e6 for s in range(1,1001)]

hyper_parameters = {"hidden":hidden_opt, "l1":l1_opt}

search_criteria = {"strategy":"RandomDiscrete",

"max_models":10, "max_runtime_secs":100,

"seed":123456}

from h2o.grid.grid_search import H2OGridSearch

model_grid = H2OGridSearch(H2ODeepLearningEstimator,

hyper_params=hyper_parameters,

search_criteria=search_criteria)

model_grid.train(x=x, y=y,

distribution="multinomial", epochs=1000,

training_frame=train_h, validation_frame=test_h,

score_interval=2, stopping_rounds=3,

stopping_tolerance=0.05,

stopping_metric="misclassification")
model_grid
model_grid.confusion_matrix
model_grid.mse()