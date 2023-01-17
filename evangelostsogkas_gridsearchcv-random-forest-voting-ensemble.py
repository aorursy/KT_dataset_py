import numpy as np

import pandas as pd

from sklearn.metrics import mean_squared_log_error

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.ensemble import RandomForestRegressor, VotingRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.tree import DecisionTreeRegressor



# Load data

df_train = pd.read_csv('/kaggle/input/inf131-2019/train.csv')

X = df_train.drop(['casual', 'registered', 'cnt', 'atemp', 'windspeed'], axis=1)

y = df_train['cnt']



# train-test split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)



# score function

def rmsle_score(y_true, y_pred):

    for i, y in enumerate(y_pred):

        if y_pred[i] < 0:

            y_pred[i] = 0

    return np.sqrt(mean_squared_log_error(y_true, y_pred))

# Random Forest

rf = RandomForestRegressor(n_jobs=-1, random_state=0)

parameters = {'n_estimators': [200, 400], 'max_depth': [15, 25]}

rf_cv = GridSearchCV(rf, parameters, cv=5, n_jobs=-1)

rf_cv.fit(X_train, y_train)

y_pred = rf_cv.predict(X_test)

    

print('Random Forest Best Parameters:', rf_cv.best_params_)

print('Random Forest RMSLE score:', rmsle_score(y_test, y_pred))

    
# KNN

knn = KNeighborsRegressor(n_jobs=-1)

parameters = {'n_neighbors': [4, 6, 8], 'weights': ['uniform', 'distance'], 'p': [1, 2]}

knn_cv = GridSearchCV(knn, parameters, cv=5, n_jobs=-1)

knn_cv.fit(X_train, y_train)

y_pred_knn = knn_cv.predict(X_test)



print('KNN Best Parameters:', knn_cv.best_params_)

print('KNN RMSLE score:', rmsle_score(y_test, y_pred_knn))



# Decision Tree

dt = DecisionTreeRegressor(random_state=0)

parameters = {'max_depth': [14, 16, 18], 'min_samples_leaf': [2, 4, 6]}

dt_cv = GridSearchCV(dt, parameters, cv=5, n_jobs=-1)

dt_cv.fit(X_train, y_train)

y_pred_dt = dt_cv.predict(X_test)



print('Decision Tree Best Parameters:', dt_cv.best_params_)

print('Decision Tree RMSLE score:', rmsle_score(y_test, y_pred_dt))
# Use best parameters found before

knn = KNeighborsRegressor(n_jobs=-1, n_neighbors=8, weights='distance', p=1)

dt = DecisionTreeRegressor(random_state=0, max_depth=16, min_samples_leaf=4)



# Voting

voting = VotingRegressor(estimators=[('knn', knn), ('dt', dt)], weights=None, n_jobs=-1)

voting.fit(X_train, y_train)

y_pred_voting = voting.predict(X_test)

    



print('Voting RMSLE score:', rmsle_score(y_test, y_pred_voting))