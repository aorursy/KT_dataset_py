import pandas as pd
import numpy as np
hit = pd.read_csv("../input/hittlers/Hitters.csv")
df = hit.copy()
df = df.dropna()
dms = pd.get_dummies(df[['League', 'Division', 'NewLeague']])
y = df["Salary"]
X_ = df.drop(['Salary', 'League', 'Division', 'NewLeague'], axis=1).astype('float64')
X = pd.concat([X_, dms[['League_N', 'Division_W', 'NewLeague_N']]], axis=1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.25, 
                                                    random_state=42)
from sklearn.svm import SVR
svr_rbf = SVR("rbf").fit(X_train, y_train)
import numpy as np
y_pred=svr_rbf.predict(X_test)
from sklearn.metrics import mean_squared_error
np.sqrt(mean_squared_error(y_test,y_pred))
svr_rbf
svr_params= {"C":[0.1,0.4,5,10,20,30,
               40,50]}
from sklearn.model_selection import GridSearchCV
svr_cv_model= GridSearchCV(svr_rbf,svr_params,cv=10)
svr_cv_model.fit(X_train,y_train)
svr_cv_model.best_params_
svr_tuned_cv_model=SVR("rbf",C=pd.Series(svr_cv_model.best_params_)[0]).fit(X_train,y_train)
y_pred=svr_tuned_cv_model.predict(X_test)
np.sqrt(mean_squared_error(y_test,y_pred))
#Thanks to https://github.com/mvahit/DSMLBC