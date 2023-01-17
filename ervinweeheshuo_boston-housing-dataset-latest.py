
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.feature_selection import SelectKBest, chi2, f_regression
from sklearn import datasets
from sklearn.feature_selection import RFE
from sklearn.ensemble import GradientBoostingRegressor 
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv('/kaggle/input/boston-housing-dataset/HousingData.csv')
df.info()



df=df.dropna(axis=0)
df
#Checking for multicollinearity
correlation_matrix = df.corr().round(2)
fig, ax = plt.subplots(figsize=(10,10))  
sns.heatmap(data=correlation_matrix,annot=True,linewidths=0.5,ax=ax)


       

#Using Simple Linear Regression (All Variables)
y=df['MEDV']
x=df.iloc[:,:12]

x
x=sm.add_constant(x)
model=sm.OLS(y,x)
results = model.fit()
results.summary()
#Using Simple Linear Regression (After Removing Variables with Multi-Colinearity) Remove INDUS and NOX
x=df.iloc[:,[0,1,3,5,6,7,8,9,10,11,12]]
x= sm.add_constant(x)
model1=sm.OLS(y,x)
results_update = model1.fit()
results_update.summary()
#Remove INDUX and NOX
x=df.iloc[:,[0,1,3,5,6,7,8,9,10,11,12]]
features_list=x.columns
features_list
#Using Sklearn to conduct linear regression on test data
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42)
model1=LinearRegression()
model1.fit(x_train,y_train)
y_pred=model1.predict(x_test)
y_pred=pd.DataFrame(y_pred)
print("train r2:",model1.score(x_train,y_train))
print("test r2:",model1.score(x_test,y_test))

# Selecting Predictor Variables with SelectKBest - f_regression

for i in range(11):
   selector_train=SelectKBest(f_regression,k=i+1)
   selector_train.fit_transform(x_train,y_train)
   cols=selector_train.get_support(indices=True)
   x_test1=x_test.iloc[:,cols]
   selector_train1=SelectKBest(f_regression,k=i+1).fit_transform(x_train,y_train)
   df1=pd.DataFrame(selector_train1)
   model1.fit(df1,y_train)
   print("r2:",i+1,model1.score(x_test1,y_test)) #Model works best with 11 features


x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42)
#Selecting Predictor Variables with Recursive Feature Elimination (Gradient Boosting Regressor)
for i in range(11,1,-1):
   model = GradientBoostingRegressor()
   rfe = RFE(model, i)
   rfe = rfe.fit(x_train, y_train) #error
   rfe_support = rfe.get_support()
   x_train=x_train.loc[:,rfe_support]
   x_test=x_test.loc[:,rfe_support]
   model.fit(x_train,y_train)
#Finding R2 using Linear Regression to Compare R2 (To Find Best Features)
   model1=LinearRegression()
   model1.fit(x_train,y_train)
   print("r2:",i,model1.score(x_test,y_test))
   
#Selecting Predictor Variables with RandomForestRegressor
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42)
for i in range(11,1,-1):
   model = RandomForestRegressor()
   rfe = RFE(model, i)
   rfe = rfe.fit(x_train, y_train) #error
   rfe_support = rfe.get_support()
   x_train=x_train.loc[:,rfe_support]
   x_test=x_test.loc[:,rfe_support]
   model.fit(x_train,y_train)
#Finding R2 using Linear Regression to Compare R2 (To Find Best Features)
   model1=LinearRegression()
   model1.fit(x_train,y_train)
   print("r2:",i,model1.score(x_test,y_test))


#Hyperparameter tuning for the Lasso Regression
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42)
from sklearn.linear_model import Lasso
lasso=Lasso()
alphas1=np.logspace(-10,1,20)
alphas = np.array([5, 0.5, 0.05, 0.005, 0.0005, 1, 0.1, 0.01, 0.001, 0.0001, 0 , 0.2,0.3,0.4,0.6,2,3,4,5,6]) 
alphas=alphas + alphas1
grid = GridSearchCV(estimator=lasso, param_grid=dict(alpha=alphas),cv=5)
grid.fit(x_test, y_test)

print(grid.best_estimator_.alpha)
print(grid.best_score_)
# Using Lasso Regression for Feature Selection
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42)
# Import Lasso
from sklearn.linear_model import Lasso

# Instantiate a lasso regressor: lasso
lasso = Lasso(alpha=6.158482110660267e-05)

# Fit the regressor to the data
lasso.fit(x_train, y_train)
# Compute and print the coefficients
lasso_coef = lasso.coef_
print(lasso_coef)

plt.plot(x_train.columns, lasso_coef)
plt.xticks(rotation=60)



# Using Gradient Booting Regressor
GBE=GradientBoostingRegressor()
GBE.fit(x_train,y_train)
GBE.score(x_test,y_test)

#Random Forest Regressor
RF=RandomForestRegressor()
RF.fit(x_train,y_train)
RF.score(x_test,y_test)
#Lasso
lasso1=Lasso(alpha=6.158482110660267e-05)
lasso1.fit(x_train,y_train)
lasso1.score(x_test,y_test)
#Ridge
from sklearn.linear_model import Ridge
print(grid.best_estimator_.alpha)
print(grid.best_score_)

ridge=Ridge(alpha=1.000000078475997)
grid = GridSearchCV(estimator=ridge, param_grid=dict(alpha=alphas),cv=5)
grid.fit(x_test, y_test)



ridge.fit(x_train,y_train)
ridge.score(x_test,y_test)


#Create a new model by selecting variables that have a p-value of less than 0.05 and use GBE
x=df.iloc[:,[0,3,4,5,6,7,8,9,10,11,12]]
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42)
GBE=GradientBoostingRegressor()
GBE.fit(x_train,y_train)
GBE.score(x_test,y_test)




