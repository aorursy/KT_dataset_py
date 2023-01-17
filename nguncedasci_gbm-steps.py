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
# Import data and split it
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
from sklearn.ensemble import GradientBoostingRegressor
gbm_model= GradientBoostingRegressor()
gbm_model.fit(X_train,y_train)
#Important Parameters
#learning_rate
#n_estimators should begin with 1000 for GBM and XG Boost
#max_depth
#subsample
#Prediction
y_pred=gbm_model.predict(X_test)
np.sqrt(mean_squared_error(y_pred,y_test))
#Model Tuning
gbm_params = {
    'learning_rate': [0.001, 0.01, 0.1, 0.2],
    'max_depth': [3, 5, 8,50,100],
    'n_estimators': [200, 500, 1000, 2000],
    'subsample': [1,0.5,0.75],
}
gbm= GradientBoostingRegressor()
gbm_cv_model= GridSearchCV(gbm,gbm_params,cv=10,n_jobs=-1,verbose=2)  #verbose gives us some information 
gbm_cv_model.fit(X_train,y_train)
gbm_cv_model.best_params_
gbm_tuned_model= GradientBoostingRegressor(learning_rate= 0.2,
                                           max_depth= 3,
                                           n_estimators=500, subsample= 0.75)
gbm_tuned_model.fit(X_train,y_train)
y_pred=gbm_tuned_model.predict(X_test)
np.sqrt(mean_squared_error(y_pred,y_test))    #test error
# We found 413 for KNN, 
#          367 for SVR,
#          363 for Artifical Neural Network.
#          376 for CART
#          349 for Bagged Trees
#          350 for Random Forest
#And now,  344 for GBM

#In these models, the best one is GBM model for "hitters" data set, till now.
# VARIABLES' IMPORTANCE LEVEL  (BONUS PART)
Importance = pd.DataFrame({"Importance": gbm_tuned_model.feature_importances_*100},
                         index = X_train.columns)
Importance.sort_values(by = "Importance", 
                       axis = 0, 
                       ascending = True).plot(kind ="barh", color = "r")

plt.xlabel("Variables' Importance Level")
