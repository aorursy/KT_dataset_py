# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import glob, re

from sklearn import *

from datetime import datetime





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/surprise-me/training_data.csv')

test = pd.read_csv('../input/surprise-me/test_data.csv')
col = [c for c in train if c not in ['id', 'air_store_id','visit_date','visitors']]

print (col)
#set up an array of models - test nearest neighbours 

model1 = neighbors.KNeighborsRegressor(n_jobs=-1, n_neighbors=2)

model2 = neighbors.KNeighborsRegressor(n_jobs=-1, n_neighbors=3)

model3 = neighbors.KNeighborsRegressor(n_jobs=-1, n_neighbors=4)

model4 = neighbors.KNeighborsRegressor(n_jobs=-1, n_neighbors=5)



#model1 = ensemble.GradientBoostingRegressor(learning_rate=0.1)

#model2 = neighbors.KNeighborsRegressor(n_jobs=-1, n_neighbors=5)

#model3 = ensemble.AdaBoostRegressor(tree.DecisionTreeRegressor(max_depth=4), 

 #                                   n_estimators=300, random_state=np.random.RandomState(1))

#model4 = tree.DecisionTreeRegressor(max_depth=7)

#model5 = ensemble.RandomForestRegressor(max_depth=5, random_state=0)

#model3 = ensemble.RandomForestRegressor(max_depth=10, random_state=0)
model1.fit(train[col], np.log1p(train['visitors'].values))

model2.fit(train[col], np.log1p(train['visitors'].values))

model3.fit(train[col], np.log1p(train['visitors'].values))

model4.fit(train[col], np.log1p(train['visitors'].values))
def RMSLE(y, pred):

    return metrics.mean_squared_error(y, pred)**0.5



#print('RMSE GradientBoostingRegressor: ', RMSLE(np.log1p(train['visitors'].values), model1.predict(train[col])))

#print('RMSE KNeighborsRegressor: ', RMSLE(np.log1p(train['visitors'].values), model2.predict(train[col])))

#print('RMSE RandomforestRegressor 10 trees: ', RMSLE(np.log1p(train['visitors'].values), model3.predict(train[col])))

#print('RMSE DecisionTreeRegressor: ', RMSLE(np.log1p(train['visitors'].values), model4.predict(train[col])))

#print('RMSE RandomForestRegressor: ', RMSLE(np.log1p(train['visitors'].values), model5.predict(train[col])))

#RMS Error on the models

print('2 neighbours: ', RMSLE(np.log1p(train['visitors'].values), model1.predict(train[col])))

print('3 neighbours: ', RMSLE(np.log1p(train['visitors'].values), model2.predict(train[col])))

print('4 neighbours: ', RMSLE(np.log1p(train['visitors'].values), model3.predict(train[col])))

print('5 neighbours: ', RMSLE(np.log1p(train['visitors'].values), model4.predict(train[col])))
#predict the test data

test['visitors'] = (model1.predict(test[col]) + model2.predict(test[col])) / 2
test['visitors'] = np.expm1(test['visitors']).clip(lower=0.)

sub1 = test[['id','visitors']].copy()
sub1.to_csv('nn-submission.csv')
#set up random forest regressors

model1 = ensemble.RandomForestRegressor(max_depth=5, random_state=0)

model2 = ensemble.RandomForestRegressor(max_depth=10, random_state=0)

model3 = ensemble.RandomForestRegressor(max_depth=15, random_state=0)
model1.fit(train[col], np.log1p(train['visitors'].values))

model2.fit(train[col], np.log1p(train['visitors'].values))

model3.fit(train[col], np.log1p(train['visitors'].values))
print('RFR: depth 5: ', RMSLE(np.log1p(train['visitors'].values), model1.predict(train[col])))

print('RFR: depth 10: ', RMSLE(np.log1p(train['visitors'].values), model2.predict(train[col])))

print('RFR: depth 15: ', RMSLE(np.log1p(train['visitors'].values), model3.predict(train[col])))
test['visitors'] = (model2.predict(test[col]) + model3.predict(test[col])) / 2

test['visitors'] = np.expm1(test['visitors']).clip(lower=0.)

sub1 = test[['id','visitors']].copy()
sub1.to_csv('rfr-submission.csv')