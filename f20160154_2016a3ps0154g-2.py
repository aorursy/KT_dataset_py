import numpy as np

import matplotlib as plt

import pandas as pd

import os
dataset=pd.read_csv("../input/train.csv")
X=dataset.iloc[:,:-1]

y=dataset.iloc[:,11]

#dataset.head()
#X=pd.get_dummies(X,columns=[],drop_first=True)

#import sys

#!{sys.executable} -m pip install xgboost
#dummy=X.corr(method='pearson')

#dummy.to_csv("/home/dell/Desktop/corcoef.csv")

dummy=X.corr().abs()

s=dummy.unstack()

so = s.sort_values(kind="quicksort")

#so.to_csv("/home/dell/assignment2/corcoef1.csv")

#X.to_csv("/home/dell/assignment2/new_trainx.csv")

%matplotlib inline

import pandas_profiling 



#pandas_profiling.ProfileReport(dataset)

#import seaborn as sns

#corr = X.corr()

#sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True), square=True)
from sklearn import linear_model

from sklearn.model_selection import train_test_split

from matplotlib import pyplot as plt

from sklearn.model_selection import KFold

#from sklearn.cross_validation import cross_val_score, cross_val_predict

from sklearn import metrics

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_squared_error

#from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import RandomForestRegressor

import xgboost

from xgboost import XGBRegressor

from sklearn.ensemble import GradientBoostingRegressor

#from sklearn.neural_network import MLPRegressor

import math
X1=X.values

y1=y.values

kf = KFold(n_splits=5)

X1=np.delete(X1,[5],axis=1)

#X['log_value'] = np.log(X['Total Volume'])

#print(df1) 

#kf.get_n_splits(X)


for train_index, test_index in kf.split(X):

 #print("TRAIN:", train_index, "TEST:", test_index)

 X_train, X_test = X1[train_index], X1[test_index]

 y_train, y_test = y1[train_index], y1[test_index]

 #X_train=np.delete(X_train,[0,1,2,36,43,21,31,40,45],axis=1)

 #X_test=np.delete(X_test,[0,1,2,36,43,21,31,40,45],axis=1)

 #sc_X=StandardScaler()

 #X_train=sc_X.fit_transform(X_train)

 #X_test=sc_X.fit_transform(X_test)

 #y_train=sc_X.fit_transform(y_train)

 #y_test=sc_X.fit_transform(y_test)

 #lm = linear_model.LinearRegression()

 #regressor = RandomForestRegressor(n_estimators=100)

 model = XGBRegressor(max_depth=15,objective="reg:linear",learning_rate=0.1,booster='gbtree')

 #regressor = MLPRegressor(hidden_layer_sizes=(100,200,300,400,300,200,100),verbose = True)

 #model = lm.fit(X_train, y_train)

 #regressor.fit(X_train, y_train)

 model.fit(X_train,y_train)

 #regressor.fit(X=X_train,y=y_train)

 #y_pred = lm.predict(X_test)

 #y_pred = regressor.predict(X_test)

 y_pred=model.predict(X_test)

 #y_pred = regressor.predict(X_test)

 #plt.scatter(y_test, predictions)

 #plt.xlabel("True Values")

 #plt.ylabel("Predictions")

 #print("Score:", (mean_squared_error(y_pred, y_test)))



 #scores = cross_val_score(model, df, y, cv=5)

 #print “Cross-validated scores:”, scores
test1=pd.read_csv("../input/test.csv")

FOODID=pd.read_csv("../input/test.csv", usecols = ['id'])

#test1=pd.get_dummies(test1,columns=[],drop_first=True)

#car_id=test1.iloc[:,0]

test2=test1.values



test2=np.delete(test2,[5],axis=1)

#pd.DataFrame(test2).dtypes

#sc_X=StandardScaler()

#test2=sc_X.fit_transform(test2)



test3=pd.DataFrame(test2)

#test3.to_csv("/home/dell/assignment2/new_testx.csv")
X_traindf=pd.DataFrame(X_train)

missing_cols = set( X_traindf.columns ) - set( test3.columns )

# Add a missing column in test set with default value equal to 0

for c in missing_cols:

    test3[c] = 0

# Ensure the order of column in the test set is in the same order than in train set

test3 = test3[X_traindf.columns]

test2=test3.values

#y_pred = regressor.predict(test2)

y_pred = model.predict(test2)
#scalery=StandardScaler()

#y_pred = scalery.fit(y_pred)

#y_pred=scalery.inverse_transform(y_pred)

y_pred1=pd.DataFrame(y_pred)

final  = pd.concat([FOODID,y_pred1], axis=1)

final.to_csv("seventh_result.csv", index=False)

#y_pred1.to_csv("/home/dell/assignment2/pred.csv")
