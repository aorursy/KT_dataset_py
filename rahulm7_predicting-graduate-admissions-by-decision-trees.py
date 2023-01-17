# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/Admission_Predict.csv')
df.head(10)
# Check null values in the dataset - pretty clean
df.isnull().sum()
df.columns
df.describe()
df_features = df.drop(['Chance of Admit ','Serial No.'],axis=1)
df_target = df['Chance of Admit ']
# Prepare training and test dataset
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(df_features,df_target,test_size=0.2,random_state=123)
print(X_train.shape,X_test.shape,Y_train.shape,Y_test.shape)
## We can see, data is not standardized, it's not required if we are building decision tress
dt = DecisionTreeRegressor(criterion='mse',max_depth=2,splitter='best',min_samples_split=2)
dt_fit = dt.fit(X_train,Y_train)
from sklearn.metrics import mean_squared_error,r2_score
y_pred = dt_fit.predict(X_test)
rmse = np.sqrt(mean_squared_error(Y_test, y_pred))
print(rmse,r2_score(Y_test,y_pred))
predictions = pd.DataFrame(y_pred,Y_test).reset_index()
predictions.columns = ['Predictions','Actual']
predictions.head(20)
dt.feature_importances_
df_features.columns
#### Most Important Features 1) CGPA 2) GRE Score
from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(dt, X_train, Y_train, cv=10)
np.mean(cv_scores)
from numpy.random import randint
params = {"max_depth":[1,5],
         "max_features":randint(1,7,size=4), # Use maximum 4 features 
         "min_samples_leaf":randint(1,4,size=3)}
from sklearn.model_selection import RandomizedSearchCV
random_search = RandomizedSearchCV(estimator = dt,param_distributions = params,n_iter=15,cv=5)
# Fit the random search on dataset
random_search.fit(df_features, df_target)
print(random_search.best_estimator_)
print(random_search.best_params_)
print(random_search.best_score_)
#### We get an improved accuracy over previous cross validation score of 67% to 72%
#### We can train a random forest regressor to improve overall accuracy