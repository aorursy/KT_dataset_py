
import pandas as pd 
from sklearn.model_selection import train_test_split, GridSearchCV,cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
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


from sklearn.ensemble import BaggingRegressor

bag_model=BaggingRegressor(bootstrap_features=True)
bag_model.fit(X_train,y_train)
bag_model.n_estimators
bag_model.estimators_   #different trees
bag_model.estimators_samples_
bag_model.estimators_features_
bag_model.estimators_[0]
import numpy as np
y_pred=bag_model.predict(X_test)
np.sqrt(mean_squared_error(y_test,y_pred))
bag_params={"n_estimators": range(2,20)}
bag_cv_model= GridSearchCV(bag_model, bag_params,cv=10)
bag_cv_model.fit(X_train,y_train)
bag_cv_model.best_params_
bag_tuned=BaggingRegressor(n_estimators= 12,random_state=45)
bag_tuned.fit(X_train,y_train)
y_pred=bag_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_pred,y_test))
# We found 413 for KNN, 
#          367 for SVR,
#          363 for Artifical Neural Network.
#          376 for CART
#And now,  349 for Bagged Trees

#In these models(KNN-SVR-ANN-CART-Bagged Trees), the best one is Bagged Trees model for "hitters" data set, till now.

