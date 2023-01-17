import random

import pandas as pd

import seaborn as sns

import numpy as np

import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor,AdaBoostRegressor,BaggingRegressor

from sklearn.gaussian_process import GaussianProcessRegressor

from sklearn.decomposition import PCA

from sklearn.pipeline import make_pipeline

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegressionCV

from sklearn.metrics import accuracy_score,mean_absolute_error

data = pd.read_csv('../input/train.csv')



x = data.drop(['target','id'], axis = 1)

y = data['target']
train_x, test_x, train_y, test_y = train_test_split(x,y)
model = LogisticRegressionCV(cv=5, verbose = 2, n_jobs = 10)

model.fit(train_x, train_y)
mean_absolute_error(test_y,model.predict(test_x))
gbr = GradientBoostingRegressor(n_estimators=25,learning_rate=0.001, max_depth=5,max_features=36,max_leaf_nodes=5)
gbr.fit(train_x,train_y)
mean_absolute_error(test_y,gbr.predict(test_x))
rf = RandomForestRegressor(n_estimators=25,max_depth=5,max_features=36,max_leaf_nodes=5)
rf.fit(train_x,train_y)
mean_absolute_error(test_y,rf.predict(test_x))
abr = AdaBoostRegressor(n_estimators=25,learning_rate=0.01)
abr.fit(train_x,train_y)
mean_absolute_error(test_y,abr.predict(test_x))
br = BaggingRegressor(n_estimators=25,n_jobs=10,max_features=25,max_samples=10)
br.fit(train_x,train_y)
mean_absolute_error(test_y,br.predict(test_x))
gpr = GaussianProcessRegressor(n_restarts_optimizer=10)
gpr.fit(train_x,train_y)
mean_absolute_error(test_y,gpr.predict(test_x))
models = make_pipeline(PCA(n_components=2),

                      LogisticRegressionCV(cv=5, verbose = 2, n_jobs = 10))
models.fit(train_x,train_y)
mean_absolute_error(test_y,models.predict(test_x))
n_comps = []

x_val = []

for i in range(1,150):

    models = make_pipeline(PCA(n_components=i),

                          LogisticRegressionCV(cv=5, n_jobs = 10))

    models.fit(train_x,train_y)

    x_val.append(i)

    n_comps.append(mean_absolute_error(test_y,models.predict(test_x)))
plt.scatter(x_val, n_comps)

plt.show()
n_comps = []

x_val = []

for i in range(1,150):

    models = make_pipeline(PCA(n_components=i),

                          AdaBoostRegressor(learning_rate=0.01))

    models.fit(train_x,train_y)

    x_val.append(i)

    n_comps.append(mean_absolute_error(test_y,models.predict(test_x)))

    

plt.scatter(x_val, n_comps)

plt.show()
n_comps = []

x_val = []

for i in range(1,150):

    models = make_pipeline(PCA(n_components=i),

                         BaggingRegressor(max_samples=10))

    models.fit(train_x,train_y)

    x_val.append(i)

    n_comps.append(mean_absolute_error(test_y,models.predict(test_x)))

    

plt.scatter(x_val, n_comps)

plt.show()
n_comps = []

x_val = []

for i in range(1,150):

    models = make_pipeline(PCA(n_components=i),

                         GradientBoostingRegressor(learning_rate=0.01))

    models.fit(train_x,train_y)

    x_val.append(i)

    n_comps.append(mean_absolute_error(test_y,models.predict(test_x)))

    

plt.scatter(x_val, n_comps)

plt.show()
n_comps = []

x_val = []

for i in range(1,150):

    models = make_pipeline(PCA(n_components=i),

                         RandomForestRegressor(max_depth = 10,n_estimators=100))

    models.fit(train_x,train_y)

    x_val.append(i)

    n_comps.append(mean_absolute_error(test_y,models.predict(test_x)))

    

plt.scatter(x_val, n_comps)

plt.show()
test = pd.read_csv('../input/test.csv')



models = make_pipeline(PCA(n_components=35),

                          RandomForestRegressor(max_depth = 10,n_estimators=100))



models.fit(train_x,train_y)

preds = models.predict(test.drop('id',axis=1))
preds
sub = pd.DataFrame({'id':test['id'],'target':preds})
sub.to_csv('sub.csv',index=False)