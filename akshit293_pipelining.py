import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
data=pd.read_csv('../input/real-estate-price-prediction/Real estate.csv',index_col='No')
data.head()
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

pipe=Pipeline([("StandardScaler",StandardScaler()),("ridge",Ridge())])
#Splitting data into train and test data
from sklearn.model_selection import train_test_split
y=pd.DataFrame()
y['Price']=data['Y house price of unit area']
data.drop('Y house price of unit area',axis=1,inplace=True)
X_train,X_test,y_train,y_test=train_test_split(data,y,random_state=0)
#fitting out pipepline with train data
pipe.fit(X_train,y_train)
#score
print(pipe.score(X_test,y_test))
print(pipe.score(X_train,y_train))

#step name with double underscore then the name of parameter to perform grid search.
param={"step2__alpha":[0.01,0.1,1,10,100]}
from sklearn.model_selection import GridSearchCV
pipe2=Pipeline([('step1',StandardScaler()),('step2',Ridge())])
grid=GridSearchCV(pipe2,param_grid=param,cv=5)
grid.fit(X_train,y_train)
print("Best cv accuracy : ",grid.best_score_)
print("Best parameter : ",grid.best_params_)
print("Train score : ",grid.score(X_train,y_train))
print("Test score : ",grid.score(X_test,y_test))
from sklearn.pipeline import make_pipeline
pipe3=make_pipeline(StandardScaler(),Ridge())

#names of steps can be seen using steps attribute
#if any step has same class, a number will be appended to its name
print(pipe3.steps)
#fitting the pipeline using train data
pipe3.fit(X_train,y_train)
print(pipe3.named_steps.keys())
print(pipe3.named_steps.values())
print(pipe3.steps)
param={"ridge__alpha":[0.01,0.1,0,1,10,100]}
grid2=GridSearchCV(pipe3,param,cv=5)
grid2.fit(X_train,y_train)
print("best estimator ",grid2.best_estimator_)
print("best param ",grid2.best_params_)
print("test score ",grid2.score(X_test,y_test))
print("train score ",grid2.score(X_train,y_train))
