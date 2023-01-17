# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv("/kaggle/input/vehicle-dataset-from-cardekho/car data.csv")
df.head()
df.isnull().sum()
print(df['Fuel_Type'].unique())
print(df['Seller_Type'].unique())
print(df['Transmission'].unique())
print(df['Owner'].unique())
df['Fuel_Type'].value_counts()
sns.barplot('Fuel_Type','Selling_Price',data=df)
df['Seller_Type'].value_counts()
sns.barplot('Seller_Type','Selling_Price',data=df)
df['Transmission'].value_counts()
sns.barplot('Transmission','Selling_Price',data=df)
df['Owner'].value_counts()
sns.barplot('Owner','Selling_Price',data=df)
sns.regplot('Selling_Price','Present_Price',data=df)
df['Current_year']=2020
df['car_age']=df['Current_year']-df['Year']
df.head()
df.drop(['Car_Name','Year','Current_year'],axis=1,inplace=True)
df.head()
data=pd.get_dummies(df,drop_first=True)
data.head()
data.shape
data.dtypes
data[['Kms_Driven','Owner','car_age','Fuel_Type_Diesel','Fuel_Type_Petrol','Seller_Type_Individual','Transmission_Manual']] =data[['Kms_Driven','Owner','car_age','Fuel_Type_Diesel','Fuel_Type_Petrol','Seller_Type_Individual','Transmission_Manual']] .astype('float64')
data.dtypes
X=data.drop('Present_Price',axis='columns')
X.head()
y=data.Present_Price
y.head()
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
from sklearn.preprocessing import StandardScaler
sc =StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
reg=LinearRegression()
reg.fit(X_train,y_train)
pred=reg.predict(X_test).round(2)
from sklearn.metrics import r2_score
scores=cross_val_score(reg,X_train,y_train,cv=5)
print('R square:', r2_score(y_test,pred))   
print('5 CV scores:' ,scores)
print('CV Scores mean',scores.mean())

compare = pd.DataFrame({'Real Values':y_test, 'Predicted Values':pred})
compare.head()

import seaborn as sns
sns.distplot(y_test-pred)
from sklearn.tree import DecisionTreeRegressor  
from sklearn.model_selection import cross_val_score
from sklearn import metrics
dtregressor = DecisionTreeRegressor()  
dtregressor.fit(X_train, y_train) 
y_pred=dtregressor.predict(X_test).round(2)

r_square=metrics.r2_score(y_test,y_pred)
print('R square:',r_square)

dtscores=cross_val_score(dtregressor,X_train,y_train,cv=5)
print('5 CV Scores:',dtscores)
print('CV Scores mean:' ,dtscores.mean())
df2 = pd.DataFrame({'Real Values':y_test, 'Predicted Values':y_pred})
df2.head()

import seaborn as sns
sns.distplot(y_test-y_pred)
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
ridge=Ridge()
parameters={'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40]}
ridge_regressor=GridSearchCV(ridge,parameters,scoring='neg_mean_squared_error',cv=5)
ridge_regressor.fit(X_train,y_train)

print(ridge_regressor.best_params_)
print(ridge_regressor.best_score_)
from sklearn.model_selection import cross_val_score
prediction=ridge_regressor.predict(X_test).round(2)
ridge=Ridge(alpha=0.01)
rcvscores=cross_val_score(ridge,X_train,y_train,cv=5)
r_score=r2_score(y_test,prediction)
print('R square:',r_score)
print('5 CV Scores:',rcvscores)
print('CV Scores mean:',rcvscores.mean())
df3 = pd.DataFrame({'Real Values':y_test, 'Predicted Values':prediction})
df3.head()

import seaborn as sns
sns.distplot(y_test-prediction)
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
lasso=Lasso()
parameters={'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40]}
lasso_regressor=GridSearchCV(lasso,parameters,scoring='neg_mean_squared_error',cv=5)

lasso_regressor.fit(X_train,y_train)
print(lasso_regressor.best_params_)
print(lasso_regressor.best_score_)
from sklearn.model_selection import cross_val_score
lasso_prediction=lasso_regressor.predict(X_test).round(2)
lasso=Lasso(alpha=1)
lcvscores=cross_val_score(lasso,X_train,y_train,cv=5)
r_score=r2_score(y_test,prediction)
print('R Square:',r_score)
print('5 CV Scores:',lcvscores)
print('CV Scores mean:',lcvscores.mean())

df4 = pd.DataFrame({'Real Values':y_test, 'Predicted Values':lasso_prediction})
df4.head()

import seaborn as sns
sns.distplot(y_test-lasso_prediction)
from sklearn.ensemble import RandomForestRegressor

rfregressor = RandomForestRegressor(n_estimators=200, random_state=0)
rfregressor.fit(X_train, y_train)
y_pred = rfregressor.predict(X_test)
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(random_state = 42)
from pprint import pprint
# Look at parameters used by our current forest
print('Parameters currently in use:\n')
pprint(rf.get_params())
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(random_state = 42)
from pprint import pprint
# Look at parameters used by our current forest
print('Parameters currently in use:\n')
pprint(rf.get_params())

from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
pprint(random_grid)
rf = RandomForestRegressor()
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 10, cv = 10,verbose=2, random_state=42, n_jobs = 1)
rf_random.fit(X_train,y_train)
rf_random.best_score_
rf_random.best_params_
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
prediction=rf_random.predict(X_test).round(2)
randomfr=RandomForestRegressor(n_estimators=2000,min_samples_split= 5,min_samples_leaf= 2,max_features='auto',
 max_depth=50,bootstrap= True)

rcvscores=cross_val_score(randomfr,X_train,y_train,cv=5)
r_score=r2_score(y_test,prediction)
print('R square:',r_score)
print('5 CV Scores:',rcvscores)
print('CV Scores mean:',rcvscores.mean())
df5 = pd.DataFrame({'Real Values':y_test, 'Predicted Values':predictions})
df5.head()