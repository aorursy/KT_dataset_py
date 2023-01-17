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
#Set and fit the model
!pip install lightgbm
from lightgbm import LGBMRegressor
lgbm_model=LGBMRegressor()
lgbm_model.fit(X_train,y_train)
# Prediction
y_pred=lgbm_model.predict(X_test,num_iteration=lgbm_model.best_iteration_)
np.sqrt(mean_squared_error(y_test,y_pred))
#Model Tuning
lgbm_model
#Important Params
#learning_rate
#n_estimators
#max_depth
lgbm_grid = {
    'colsample_bytree': [0.4, 0.5,0.6,0.9,1],
    'learning_rate': [0.01, 0.1, 0.5,1],
    'n_estimators': [20, 40, 100, 200, 500,1000],
    'max_depth': [1,2,3,4,5,6,7,8] }

lgbm = LGBMRegressor()
lgbm_cv_model = GridSearchCV(lgbm, lgbm_grid, cv=10, n_jobs = -1, verbose = 2)
lgbm_cv_model.fit(X_train,y_train)
lgbm_cv_model.best_params_
lgbm_tuned = LGBMRegressor(learning_rate = 0.1, 
                           max_depth = 5, 
                           n_estimators = 40,
                          colsample_bytree = 0.4)

lgbm_tuned = lgbm_tuned.fit(X_train,y_train)
y_pred= lgbm_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))
# We found 413 for KNN, 
#          367 for SVR,
#          363 for Artifical Neural Network.
#          376 for CART
#          349 for Bagged Trees
#          350 for Random Forest
#          344 for GBM
#          355 for XG Boosting
#And now,  377 for Light GBM

#In these models, the best one is GBM model for "hitters" data set, till now.