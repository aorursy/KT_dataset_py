# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import missingno as msno
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV
from sklearn.metrics import mean_squared_error, r2_score
df = pd.read_csv("/kaggle/input/hitters/Hitters.csv") #reads the file
df.head() #returns the first 5 values
df.info() 
df.isnull().sum() #returns the number of null values in columns
msno.bar(df);  # visualizing missing values
df = df.dropna() 
dms = pd.get_dummies(df[["League", "Division", "NewLeague"]])
y = df[["Salary"]]
X_ = df.drop(["Salary", "League", "Division", "NewLeague"], axis = 1).astype("float64")
X = pd.concat([X_, dms[["League_N", "Division_W", "NewLeague_N"]]], axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)
X.head()
y.head()
ridge = Ridge().fit(X_train, y_train) #we created a model
ridge
ridge.coef_ #coefficients
ridge.intercept_ #fixed value - b0
y_pred = ridge.predict(X_train) #we guessed with X_train.
y_pred[:10] 
#train error
#RMSE = Root Mean Square Error
RMSE = np.sqrt(mean_squared_error(y_train, y_pred))
RMSE
# RMSE with cross validation
np.sqrt(np.mean(-cross_val_score(ridge, X_train, y_train, cv = 10, scoring ="neg_mean_squared_error")))

#test error
y_test_pred = ridge.predict(X_test) # we guessed with X_test.
RMSE = np.sqrt(mean_squared_error(y_test, y_test_pred))
RMSE
alphas = 10**np.linspace(10,-2,100)*0.5
ridgeCV = RidgeCV(alphas = alphas, scoring = "neg_mean_squared_error", cv = 10, normalize = True)
ridgeCV.fit(X_train, y_train)
ridgeCV.alpha_ # optimum alpha value
ridge_tuned = Ridge(alpha = ridgeCV.alpha_).fit(X_train, y_train)
y_pred = ridge_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))
lasso = Lasso().fit(X_train, y_train)#we created a model
lasso
lasso.coef_ ##coefficients
lasso.intercept_ #fixed value - b0
lasso.predict(X_train)[:10] #we guessed with X_train.
lasso.predict(X_test)[:10] #we guessed with X_test.
#test error
y_pred = lasso.predict(X_test)
RMSE = np.sqrt(mean_squared_error(y_test, y_pred)) 
RMSE
alphas =  np.random.randint(0,1000,100)
lassoCV = LassoCV(alphas = alphas, cv = 10, max_iter = 100000).fit(X_train, y_train)
lassoCV.alpha_ # optimum alpha value
final_lasso = Lasso(alpha = lassoCV.alpha_).fit(X_train, y_train)
y_pred = final_lasso.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))
em = ElasticNet().fit(X_train, y_train)
em.coef_
em.intercept_
em.predict(X_train)[:10]
em.predict(X_test)[:10]
y_pred = em.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))
r2_score(y_test, y_pred)
alphas = 10**np.linspace(10,-2,100)*0.5
emCV = ElasticNetCV(alphas = alphas, cv = 10).fit(X_train, y_train)
emCV.alpha_
elanet_tuned = ElasticNet(alpha = emCV.alpha_).fit(X_train, y_train)
y_pred = elanet_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))