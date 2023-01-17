import pandas as pd

import numpy as np



df = pd.read_csv("../input/80-cereals/cereal.csv").copy()
df.head()
df.name.unique()
df.isnull().sum()
df.describe().T
dummies = df['name'].str.get_dummies()
dummies
df = pd.concat([df,dummies],axis=1)
df.head()
df=df.drop(['name'],axis=1)
dummies_mfr = df['mfr'].str.get_dummies()
df=pd.concat([df,dummies_mfr],axis=1)
df = df.drop(['mfr'],axis=1)
df.head()
dummies_c = df['type'].str.get_dummies()
df = pd.concat([df,dummies_c],axis=1)
df = df.drop(['type'],axis=1)
df.head()
from sklearn.model_selection import train_test_split
y = df.rating

x = df.drop('rating',axis=1)
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=42)
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import mean_squared_error
gbm_model = GradientBoostingRegressor()

gbm_model.fit(X_train, y_train)

y_pred = gbm_model.predict(X_test)

np.sqrt(mean_squared_error(y_test, y_pred))
gbm_model.score(X_test,y_test)
gbm_params = {

    'learning_rate': [0.001, 0.01, 0.1, 0.2],

    'max_depth': [3, 5, 8,50,100],

    'n_estimators': [200, 500, 1000, 2000],

    'subsample': [1,0.5,0.75],

}
from sklearn.model_selection import train_test_split, GridSearchCV,cross_val_score
gbm = GradientBoostingRegressor()

gbm_cv_model = GridSearchCV(gbm, gbm_params, cv = 10, n_jobs = -1, verbose = 2)

gbm_cv_model.fit(X_train, y_train)
gbm_cv_model.best_params_
gbm_tuned = GradientBoostingRegressor(learning_rate = 0.1,  

                                      max_depth = 50, 

                                      n_estimators = 1000, 

                                      subsample = 0.5)



gbm_tuned = gbm_tuned.fit(X_train,y_train)
y_pred = gbm_tuned.predict(X_test)

np.sqrt(mean_squared_error(y_test, y_pred))
from sklearn.metrics import accuracy_score
gbm_tuned.score(X_test, y_test)#test accuracy score