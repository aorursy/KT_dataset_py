# Importing all the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Loading the files
train = pd.read_csv("../input/slashing-the-lowest-prices/Train.csv")
test = pd.read_csv("../input/slashing-the-lowest-prices/Test.csv")
# Printing them
train.head()
test.head()
# Changing the type
test['Demand'] = test['Demand'].astype('int64')

# # Changing the type
train['Demand'] = train['Demand'].astype('int64')
# Fixing the X_train and dropping columns
X_train = train.drop(['Item_Id', 'Date', 'Low_Cap_Price'],axis = 1)
X_train.columns
# Same process like above
X_test = test.drop(['Item_Id', 'Date'], axis = 1)
X_test.columns
# Setting the predict variabkle here and the shape
y_train = train['Low_Cap_Price']
X_train.shape
# Trying LR
from sklearn.linear_model import LinearRegression
clf = LinearRegression()
clf.fit(X_train, y_train)
preds = clf.predict(X_test)
preds
#submission = pd.DataFrame({'Item_Id':test['Item_Id'], 'Low_Cap_Price':preds})
#submission.to_csv('SumissionFile.csv', index=False)
# Trying RF

from sklearn.ensemble import RandomForestRegressor
clf2 = RandomForestRegressor(bootstrap= True,max_depth=10, max_features=6, n_estimators=400,min_samples_leaf=6
                             ,min_samples_split=12, random_state=10,ccp_alpha=0,n_jobs=-1) 
clf2.fit(X_train,y_train)
preds2 = clf2.predict(X_test)
preds2
#submission = pd.DataFrame({'Item_Id':test['Item_Id'], 'Low_Cap_Price':preds2})
#submission.to_csv('SubmissionFile.csv', index=False)
# Trying Standart scaler
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(X_train)
rescaled_X_train = scaler.transform(X_train)
# Trying Knn
from sklearn.neighbors import KNeighborsRegressor
clf3 = KNeighborsRegressor(algorithm='brute')

clf3.fit(X_train,y_train)
#from sklearn.neighbors import KNeighborsRegressor


preds3 = clf3.predict(X_test)
preds3
#submission = pd.DataFrame({'Item_Id':test['Item_Id'], 'Low_Cap_Price':preds3})
#submission.to_csv('Submission Files.csv', index=False)
# trying svr
from sklearn.svm import SVR
clf4 = SVR(kernel = 'linear')
clf4.fit(X_train,y_train)

preds4 = clf4.predict(X_test)
preds4
#submission = pd.DataFrame({'Item_Id':test['Item_Id'], 'Low_Cap_Price':preds4})
#submission.to_csv('Submission Files.csv', index=False)
#submission = pd.DataFrame({'Item_Id':test['Item_Id'], 'Low_Cap_Price':preds5})
#submission.to_csv('Submission Files.csv', index=False)
# Selecting the best params

from sklearn.model_selection import GridSearchCV
# Create the parameter grid based on the results of random search 
param_grid = {
    'bootstrap': [True],
    'max_depth': [10,50,80, 90, 100, 110],
    'max_features': [2, 3,5,6],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200,400, 300, 1000]
}
# Create a based model
rf = RandomForestRegressor()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 10, n_jobs = -1, verbose = 2)
grid_search.fit(X_train, y_train)
grid_search.best_params_

# The settings by which we got the best result

#from sklearn.ensemble import RandomForestRegressor
#clf2 = RandomForestRegressor(bootstrap= True,max_depth=10, max_features=6, n_estimators=400,min_samples_leaf=5
                             #,min_samples_split=12, random_state=10) 
#clf2.fit(X_train,y_train)

#This got me 99.84988 