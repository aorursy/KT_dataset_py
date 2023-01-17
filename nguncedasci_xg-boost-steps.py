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
# Import and split 
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
!pip install xgboost
import xgboost as xgb
#We can use special data structure for XGBoost to have high performanced result.

DM_train=xgb.DMatrix(data=X_train, label=y_train)   #enter dependent variable for "label"
DM_test=xgb.DMatrix(data=X_test, label=y_test)  
from xgboost import XGBRegressor
xgb_model=XGBRegressor().fit(X_train,y_train) # i prefer to use classic data structure which i get accustomed to use
#Prediction
y_pred=xgb_model.predict(X_test)
np.sqrt(mean_squared_error(y_pred,y_test))
#Model Tuning
xgb_model
# Important params
#booster
#colsample_bytree
#learning_rate  :avoids overfitting. 
#max_depth      
#n_estimators
xgb_grid = {
     'colsample_bytree': [0.4, 0.5,0.6,0.9,1], 
     'n_estimators':[100, 200, 500, 1000],
     'max_depth': [2,3,4,5,6],
     'learning_rate': [0.1, 0.01, 0.5]
}
xgb_cv=GridSearchCV(xgb_model, xgb_grid, cv=10,n_jobs=-1, verbose=2)
xgb_cv.fit(X_train,y_train)
xgb_cv.best_params_
# Final tuned model
xgb_tuned= XGBRegressor(colsample_bytree=0.6,
 learning_rate=0.1,
 max_depth= 2,
 n_estimators=1000)
xgb_tuned=xgb_tuned.fit(X_train,y_train)
y_pred=xgb_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test,y_pred))
# We found 413 for KNN, 
#          367 for SVR,
#          363 for Artifical Neural Network.
#          376 for CART
#          349 for Bagged Trees
#          350 for Random Forest
#          344 for GBM
#And now,  355 for XG Boosting

#In these models, the best one is GBM model for "hitters" data set, till now.
