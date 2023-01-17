

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import ensemble, tree, linear_model

from sklearn.preprocessing import Imputer

from sklearn.feature_selection import SelectKBest

from sklearn.decomposition import PCA

#pd.DataFrame(result_pd_RF).to_csv('resultRF.csv',index = False)

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import  MinMaxScaler

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import RandomizedSearchCV

import os

%matplotlib inline 

import pylab 

import scipy.stats as stats

from sklearn.cross_validation import train_test_split, cross_val_score

from sklearn.metrics import r2_score, mean_squared_error

from sklearn.utils import shuffle

plt.style.use('ggplot')

#This line force the graph print out in this jupyter Notebook
testset = pd.read_csv('../input/test.csv')

trainset = pd.read_csv('../input/train.csv')

trainset.isnull().sum()

## Some features have almost all null in every rows, so I will remove null> 1000 for cleaning the features a little bit.

features = trainset.columns.values

remove_features = trainset.columns[trainset.isnull().sum()>1000]

testset = testset.drop(remove_features,1)

testset_id = testset['Id']

testset = testset.drop('Id', 1)

## GarageCars has a higher correlation, so Garage Area and GarageYearBlt will be dropped.

## TotalBsmtSF has a higher correlation with 1stFlrSF, so it will be dropped as well.

trainset = trainset.drop(['1stFlrSF','GarageArea','GarageYrBlt'],1)

testset = testset.drop(['1stFlrSF','GarageArea','GarageYrBlt'],1)

trainset['SalePrice'] =np.log(trainset['SalePrice'])

train_labels = trainset['SalePrice']

trainset = trainset.drop('SalePrice',1)

from sklearn.model_selection import train_test_split

testset_index = range(len(trainset),len(trainset) + len(testset))



trainset_index, validset_index = train_test_split(range(len(trainset)),

                                    random_state = 42 , test_size = 0.3)

masterset = pd.concat([trainset,testset], axis = 0)

masterset_backup = masterset

# Getting Dummies from all other categorical vars

for col in masterset.dtypes[masterset.dtypes == 'object'].index:

    for_dummy = masterset.pop(col)

    masterset = pd.concat([masterset, pd.get_dummies(for_dummy, prefix=col)], axis=1)

    train_labels_full = train_labels

train_features_full = masterset.iloc[range(len(trainset))]

train_features = masterset.iloc[trainset_index] 

test_features = masterset.iloc[testset_index]

valid_features = masterset.iloc[validset_index]



## Take a copy of it will be a reference of train_labels

valid_labels = train_labels[validset_index].copy() 

train_labels = train_labels[trainset_index]



## RandomForest

from sklearn.ensemble import RandomForestRegressor

## Model

RF = RandomForestRegressor(n_jobs = 4, random_state = 42 )

PipeRF = Pipeline([

        ('std', MinMaxScaler()),

        ('RF', RF)

    ])

# estimator parameters

n_estimators = [1000]

components = []



minsamplesleaf = [2]

max_features =[0.4]

max_depth = [13]

param_grid_RF={'RF__n_estimators' : n_estimators,

               'RF__max_depth' : max_depth,

               'RF__max_features' : max_features,

               'RF__min_samples_leaf' : minsamplesleaf         

              }



# set model parameters to grid search object

gridCV_RF = GridSearchCV(estimator = PipeRF, 

                         param_grid = param_grid_RF,

                         scoring = 'neg_mean_squared_error',

                         cv = 3)

        

# train the model

gridCV_RF.fit(train_features.fillna(-1), train_labels)

print(gridCV_RF.best_params_)

def error_plot(actual, predict, title):

    fig = plt.figure(figsize=(8, 4))

    ax1 = fig.add_subplot(1,2,1)    

    plt.scatter(range(1,len(actual) + 1),actual-predict, s=20)

    plt.title('Sales Price Error '+ title)    

    plt.ylabel('Sales Price Error(log Actual - log Predict)')



    ax2 = fig.add_subplot(1,2,2)

    stats.probplot(actual - predict, dist="norm", plot=pylab)

    pylab.show()

## Transform neg_mean_squared_error to the metrix mean root log error



print('Score: ', (gridCV_RF.best_score_ * -1) ** 0.5)
model_RF = gridCV_RF.best_estimator_.fit(train_features_full.fillna(-1), train_labels_full)


result_RF = model_RF.predict(train_features_full.fillna(-1))

result_RF_test = model_RF.predict(test_features.fillna(-1))
error_plot(train_labels_full, result_RF, 'Error Predict vs Actual(log scale)')
from sklearn.linear_model import LinearRegression

linear = LinearRegression()

linear.fit(np.log( result_RF.reshape([-1,1]) ), train_labels_full)
result_RF = linear.predict(np.log(result_RF_test.reshape([-1,1])))
#result_RF = gridCV_RF.best_estimator_.predict(test_features.fillna(-1))

## Output result, match the require format

result_pd= pd.DataFrame(np.array(list(zip(testset_index,result_RF))),

                        columns = ['Id','SalePrice']) ## Id start from 1

result_pd.Id = result_pd.Id.astype('int') + 1 ## Id = index + 1

result_pd['SalePrice'] = result_pd['SalePrice'].apply(lambda x: np.e**(x))



pd.DataFrame(result_pd).to_csv('resultRF.csv',index = False)

result_pd[0:5]