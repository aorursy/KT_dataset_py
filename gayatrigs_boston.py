from sklearn.datasets import load_boston

dataset=load_boston()
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline
boston=pd.DataFrame(dataset.data , columns=dataset.feature_names)
boston.head()
boston['MEDV']=dataset.target
boston.head()
boston.info()
boston.nunique()
boston.describe()
boston.isnull().sum()
boston['MEDV'].plot()

from scipy.stats import skew
boston['MEDV'].skew()
(np.log1p(boston['MEDV'])).skew()
corr_matrix=boston.corr().round(2)
plt.figure(figsize=(10,7))
sns.heatmap(data=corr_matrix , annot=True)
plt.figure(figsize=(20,5))

features = ['LSTAT','RM']
target= boston['MEDV']
for i,col in enumerate(features):
    plt.subplot(1,len(features),i+1)
    x=boston[col]
    y=target
    plt.scatter(x,y, marker='o')
    plt.title(col)
    plt.xlabel(col)
    plt.ylabel('MEDV')
    
 

X=pd.DataFrame(np.c_[boston['LSTAT'],boston['RM']],columns=['LSTAT','RM'])
Y=boston['MEDV']
from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=5)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score

lin_model=LinearRegression()
lin_model.fit(X_train,Y_train)
y_test_predictv=lin_model.predict(X_test)

rmse=np.sqrt(mean_squared_error(y_test_predictv,Y_test))

r2=r2_score(y_test_predictv,Y_test)

print('the values predicted has')
print('RMSE = {}'.format(rmse))
print('r2 Score= {}'.format(r2))

# when all the columns are taken
X=boston.drop(columns=['MEDV','RAD'])
#X=pd.DataFrame(np.c_[boston['LSTAT'],boston['RM']],columns=['LSTAT','RM'])
Y=boston['MEDV']
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=7)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

pipe = make_pipeline(StandardScaler(), LinearRegression())
pipe.fit(X_train,Y_train)
y_test_predict=pipe.predict(X_test)

rmse=np.sqrt(mean_squared_error(y_test_predict,Y_test))

r2=r2_score(y_test_predict,Y_test)

print('the predicted values has')
print('RMSE={}'.format(rmse))
print('r2 score={}'.format(r2))
model_lin=LinearRegression()
model_lin.fit(X_train,Y_train)

y_test_predictv=model_lin.predict(X_test)

rmse=np.sqrt(mean_squared_error(y_test_predictv,Y_test))

r2=r2_score(y_test_predictv,Y_test)

print('the predicted values has')
print('RMSE={}'.format(rmse))
print('r2 score={}'.format(r2))
             
#LASSO
from sklearn.linear_model import Lasso

lasso_model=Lasso(alpha=0.01)
lasso_model.fit(X_train,Y_train)
L=lasso_model.predict(X_test)

rmse=(np.sqrt(mean_squared_error(Y_test,L)))

r2=r2_score(Y_test,L)

print('the lasso model has')
print('RMSE={}'.format(rmse))
print('r2 score={}'.format(r2))

from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV

l_estimator=Lasso()
parameters={'alpha':[0.01,0.1,0.3,0.5,0.9,1,4,7,9,10],
             'fit_intercept':[True,False]}
grid=GridSearchCV(estimator=l_estimator,param_grid=parameters,cv=2,n_jobs=-1)
grid.fit(X_train,Y_train)
grid.best_params_
grid.best_score_
lasso_model=Lasso(alpha=0.3)

lasso_model.fit(X_train,Y_train)

L=lasso_model.predict(X_test)

rmse=np.sqrt(mean_squared_error(Y_test,L))

r2=r2_score(Y_test,L)

print('rmse={}'.format(rmse))
print('r2 score={}'.format(r2))
from sklearn.linear_model import Ridge

R_model=Ridge(alpha=0.1)
R_model.fit(X_train,Y_train)

y_test_predict=R_model.predict(X_test)

rmse=np.sqrt(mean_squared_error(Y_test,y_test_predict))
r2=r2_score(Y_test,y_test_predict)

print('RMSE={}'.format(rmse))
print('r2 score={}'.format(r2))
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

r_estimator=Ridge()

parameters={'alpha':[0.01,0.1,0.3,0.5,0.8,10,11,12],
           'fit_intercept':[True,False]}
grid=GridSearchCV(estimator=r_estimator,param_grid=parameters,cv=2,n_jobs=-1)
grid.fit(X_train,Y_train)
grid.best_params_
r_model=Ridge(alpha=12)

r_model.fit(X_train,Y_train)

y_test_predict=r_model.predict(X_test)

rmse=np.sqrt(mean_squared_error(Y_test,y_test_predict))

r2=r2_score(Y_test,y_test_predict)

print('RMSE={}'.format(rmse))

print('r2 score={}'.format(r2))
from sklearn.linear_model import ElasticNet

e_model=ElasticNet()

e_model.fit(X_train,Y_train)

y_test_predict=e_model.predict(X_test)

rmse=np.sqrt(mean_squared_error(Y_test,y_test_predict))

r2=r2_score(Y_test,y_test_predict)

print('RMSE={}'.format(rmse))

print('r2 score={}'.format(r2))
e_estimator=ElasticNet()

parameters={'alpha':[0.01,0.1,0.3,0.5,0.8,10,11,12],
          'l1_ratio':[0.01,0.1,0.3,0.4,0.8,10,11],
            'fit_intercept':[True,False]}
grid=GridSearchCV(estimator=e_estimator,param_grid=parameters,cv=2,n_jobs=-1)
grid.fit(X_train,Y_train)
grid.best_params_
from sklearn.linear_model import ElasticNet

e_model=ElasticNet(alpha=0.1,l1_ratio=0.1)

e_model.fit(X_train,Y_train)

y_test_predict=e_model.predict(X_test)

rmse=np.sqrt(mean_squared_error(Y_test,y_test_predict))

r2=r2_score(Y_test,y_test_predict)

print('RMSE={}'.format(rmse))

print('r2 score={}'.format(r2))