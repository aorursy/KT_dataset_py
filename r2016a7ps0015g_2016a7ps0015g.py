# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



#import os

#print(os.listdir("../input"))

import matplotlib.pyplot as plt

%matplotlib inline

#from xgboost import XGBClassifier

from sklearn.ensemble import RandomForestRegressor

data = pd.read_csv('../input/train.csv')

data.head()



from sklearn.preprocessing import MinMaxScaler

data_req = data

test = pd.read_csv('../input/test.csv')

X_test = test

#X_test = X_test.drop(['XLarge Bags','Small Bags','Large Bags'],axis =1

X_test = X_test.drop(['XLarge Bags','Small Bags','Large Bags'],axis =1)

X_train = data_req.drop(['AveragePrice','XLarge Bags','Small Bags','Large Bags'],axis = 1)

X_train1 = X_train

Y_train = data_req[['AveragePrice']]

scaler = MinMaxScaler(feature_range=(0,2))             #Instantiate the scaler

X_train1[['Total Volume','4046','4225','4770','Total Bags','year']] = scaler.fit_transform(X_train1[['Total Volume','4046','4225','4770','Total Bags','year']])     #Fit and transform the data

#scaled_X_train = scaler.fit_transform(X_test)

X_train1

X_test1 = X_test

X_test1[['Total Volume','4046','4225','4770','Total Bags','year']] = scaler.fit_transform(X_test1[['Total Volume','4046','4225','4770','Total Bags','year']])     #Fit and transform the data

X_test1



from sklearn.ensemble import GradientBoostingRegressor



regressor_randomforest = RandomForestRegressor(n_estimators = 300, random_state = 0 ) 

regressor_randomforest.fit(X=X_train,y=Y_train)



from sklearn.model_selection import GridSearchCV

#TODO

#from sklearn.model_selection import GridSearchCV

from sklearn.metrics import make_scorer

from sklearn.metrics import r2_score





#TODO

clf = GradientBoostingRegressor(n_estimators = 100)        #Initialize the classifier object



parameters = {'loss':['ls', 'lad', 'huber', 'quantile'],'learning_rate':[0.01,0.05,0.1,0.5,1]}    #Dictionary of parameters



scorer = make_scorer(r2_score)         #Initialize the scorer using make_scorer



grid_obj = GridSearchCV(clf,parameters,scoring=scorer)         #Initialize a GridSearchCV object with above parameters,scorer and classifier



grid_fit = grid_obj.fit(X_train1,Y_train)        #Fit the gridsearch object with X_train,y_train



best_clf = grid_fit.best_estimator_         #Get the best estimator. For this, check documentation of GridSearchCV object



unoptimized_predictions = (clf.fit(X_train, Y_train)).predict(X_test1)      #Using the unoptimized classifiers, generate predictions

optimized_predictions = best_clf.predict(X_test1)        #Same, but use the best estimator



#acc_unop = r2_score(te_Y, unoptimized_predictions)*100       #Calculate accuracy for unoptimized model

#acc_op = r2_score(te_Y, optimized_predictions)*100         #Calculate accuracy for optimized model



#print("Accuracy score on unoptimized model:{}".format(acc_unop))

#print("Accuracy score on optimized model:{}".format(acc_op))



y_pred = regressor_randomforest.predict(X_test)



#rmse,r2,explained_var_score = performance_metrics(Y_train,y_pred)

#y_pred = optimized_predictions

y_pred



from itertools import zip_longest

import csv

final = [test['id'],y_pred]

export_data = zip_longest(*final, fillvalue = '')

with open('out.csv','w',encoding='ISO-8859-1',newline='') as myfile:

    wr = csv.writer(myfile,quoting = csv.QUOTE_ALL)

    wr.writerow(('id','AveragePrice'))

    wr.writerows(export_data)

# Any results you write to the current directory are saved as output.