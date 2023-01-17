import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split, GridSearchCV,cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale 
from sklearn import model_selection
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import BaggingRegressor

from warnings import filterwarnings
filterwarnings('ignore')
#import and split
hit = pd.read_csv("../input/hittlers/Hitters.csv")
df = hit.copy()
df = df.dropna()
dms = pd.get_dummies(df[['League', 'Division', 'NewLeague']])
y = df["Salary"]
X_ = df.drop(['Salary', 'League', 'Division', 'NewLeague'], axis=1).astype('float64')
X = pd.concat([X_, dms[['League_N', 'Division_W', 'NewLeague_N']]], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.25, 
                                                    random_state=42)
!pip install catboost
from catboost import CatBoostRegressor
catboost_model=CatBoostRegressor()
catboost_model.fit(X_train,y_train)
#Prediction
y_pred=catboost_model.predict(X_test)
np.sqrt(mean_squared_error(y_pred,y_test))
#Model Tuning
# important params
#iterations
#learning_rate
#depth
catboost_grid = {
    'iterations': [200,500,1000,2000],
    'learning_rate': [0.01, 0.03, 0.05, 0.1],
    'depth': [3,4,5,6,7,8] }
catboost=CatBoostRegressor()
catboost_cv_model=GridSearchCV(catboost, catboost_grid,cv=5, n_jobs=-1,verbose=2)
catboost_cv_model.fit(X_train,y_train)
catboost_cv_model.best_params_
catboost_tuned = CatBoostRegressor(iterations = 1000, 
                               learning_rate = 0.1, 
                               depth = 5)

catboost_tuned = catboost_tuned.fit(X_train,y_train)
y_pred = catboost_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))
# We found 413 for KNN, 
#          367 for SVR,
#          363 for Artifical Neural Network.
#          376 for CART
#          349 for Bagged Trees
#          350 for Random Forest
#          344 for GBM
#          355 for XG Boosting
#          377 for Light GBM
#And now,  356 for Cat Boost

#In these nonlinear regression models, the best one is GBM model for "hitters" data set...
