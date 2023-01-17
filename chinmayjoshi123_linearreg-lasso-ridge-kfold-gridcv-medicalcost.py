# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns
df=pd.read_csv('/kaggle/input/insurance/insurance.csv')
df.head(1)
for i in ['sex','smoker','region']:

    sns.countplot(df[i])

    plt.show()

    
for i in ['age','bmi']:

    sns.distplot(df[i])

    plt.show()
sns.distplot(df['charges'])
df.isnull().sum()
sns.pairplot(df)
sns.heatmap(df.corr(),annot=True)
df.corr()['charges']
df.head()
dfdumm=pd.get_dummies(df,drop_first=True)
dfdumm.drop('charges',axis=1,inplace=True)
dff=pd.concat([df,dfdumm],axis=1)
dff.drop(columns=['sex','smoker','region'],inplace=True)
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split



lr=LinearRegression()



X=np.array(dff.drop('charges',axis=1))

y=np.array(dff['charges']).reshape(-1,1)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1)

print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)
model=lr.fit(X_train,y_train)

y_pred=model.predict(X_test)
model.score(X_test,y_test)
from sklearn.metrics import r2_score

r2_score(y_test,y_pred)
y_pred=pd.DataFrame(y_pred)
y_test
from sklearn.metrics import r2_score,mean_squared_error



print('r2 score: ',r2_score(y_test,y_pred))



rmse=np.sqrt(mean_squared_error(y_test,y_pred))

print('rmse: ',rmse)

print('y intercept: ',lr.intercept_)

print('y coefficients: ',lr.coef_)
fig,axes=plt.subplots(figsize=(15,5))

sns.heatmap(dff.corr(),annot=True,ax=axes)

plt.show()
dff.corr()['charges']
from sklearn.linear_model import Lasso,Ridge

from sklearn.model_selection import GridSearchCV
### normal k fold 

from sklearn.model_selection import KFold

kf = KFold(n_splits=10)

print(kf)



###stratified K fold

from sklearn.model_selection import StratifiedKFold

folds = StratifiedKFold(n_splits=5)

print(folds)
from sklearn.model_selection import cross_val_score



from sklearn.linear_model import Lasso,Ridge
# param={'alpha':[0.01,0.05,0.1,0.2,0.3,0.4]}

ridge_model=Ridge()

for i in [0.01,0.05,0.1,0.2,0.3,0.4]:

    print('Ridge 10 fold results for ',i,':','\n',cross_val_score(Ridge(alpha=i), X, y,cv=10))

    print('mean of each iteration: ',np.mean(cross_val_score(Ridge(alpha=i), X, y,cv=10)))

    print('\n')



print('\n')

##Lasso using 

lasso_model=Lasso()

for i in [0.01,0.05,0.1,0.2,0.3,0.4]:

    print('Lasso 10 fold results for ',i,':','\n',cross_val_score(Lasso(alpha=i), X, y,cv=10))

    print('mean of each iteration: ',np.mean(cross_val_score(Lasso(alpha=i), X, y,cv=10)))

    print('\n')

param={'alpha':[0.01,0.05,0.1,0.2,0.3,0.4]}

ridge_model=Ridge()



model_cv=GridSearchCV(estimator=ridge_model,param_grid=param,cv=10,return_train_score=True,verbose=1)

model_cv.fit(X_train,y_train)



print(model_cv.best_estimator_)



print(model_cv.best_params_)



Y_pred=model_cv.predict(X_test)

print(r2_score(y_test,Y_pred))
''' 

so,our best hyperparameters for passing to Ridge regression model using grid search cv is - 'alpha':0.4

which is giving us the best possible accuracy from our ridge regression model,with induced penalty of 0.4

with an accuracy of 0.7622.This simply means that ridge regression model is capable of giving of this much accuracy with changes in

hyperparameter / hyperparam variation.

We can also change the X & y,to check if accuracy changes using gridsearch CV to give us more better accuracy by dropping some unwanted 

columns which are capable of inducing multicollinearity.

'''
param={'alpha':[0.01,0.05,0.1,0.2,0.3,0.4]}

lasso_model=Lasso()



model_cv=GridSearchCV(estimator=lasso_model,param_grid=param,cv=10,return_train_score=True,verbose=1)

model_cv.fit(X_train,y_train)



print(model_cv.best_estimator_)



print(model_cv.best_params_)



Y_pred=model_cv.predict(X_test)

print(r2_score(y_test,Y_pred))
''' 

so,our best hyperparameters for passing to lasso regression model using grid search cv is - 'alpha':0.01

which is giving us the best possible accuracy from our ridge regression model,with induced penalty of 0.01

with an accuracy of 0.7623.This simply means that ridge regression model is capable of giving of this much accuracy with changes in

hyperparameter / hyperparam variation.

We can also change the X & y,to check if accuracy changes using gridsearch CV to give us more better accuracy by dropping some unwanted 

columns which are capable of inducing multicollinearity.

'''