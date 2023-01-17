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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import dataset
sd= pd.read_csv('../input/Salary_Data_.csv')
#getting basic information on dataset - 1
sd.info()
#getting basic information on dataset - 2
sd.describe()
#preparing X and Y for model building
X = sd.iloc[:,: 1].values
y = sd.iloc[:,-1].values
print(X.shape)
print(y.shape)
#Checking the linear relationship between X and Y using scatter plot
colors = ('r')
plt.scatter(X,y,c=colors)
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.title("Experience vs Salary")
#Checking for Outliers
sd.boxplot(column='Salary')
#Creating a new target column to comparison
target = sd['Salary']
target.head()
import seaborn as sns
sns.distplot(target,hist=True)
#Creating a new target column(Log Transformation) to comparison
target_log = np.log(target)
sns.distplot(target_log,hist=True)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
#creating Train & Test Split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)
#Assigning Linear Model to new name
lr = LinearRegression()
#Fitting linear Regression in Train model
lr.fit(X_train,y_train)
#Creating Y_Predication using Linear model with X_test
y_pred = lr.predict(X_test)
plt.scatter(y_test,y_pred)
#Creating scatter plot to view the Actual Vs Predicted
plt.scatter(X_test,y_test, s=70,color='y', label='Actual')
plt.plot(X_train, lr.predict(X_train), color = 'b',label='Predicted')
plt.xlabel('Years of Exp')
plt.ylabel('salary')
plt.legend();
plt.grid()
plt.show()
#Viewing the Coeffecient & Intercept value
print(lr.coef_,lr.intercept_)
lr.score(X_train,y_train)
#importing Metrics from Sklearn library
from sklearn import metrics
Rsquare = metrics.r2_score(y_test,y_pred)
Rsquare
#Calculating RMSE value manually
actual_vs_pred = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
actual_vs_pred
#getting residual/error value
error = [actual_vs_pred['Actual'] - actual_vs_pred['Predicted']]
#converting into DataFrame
error = pd.DataFrame(error)
error = error.T
#Calculating RMSE value
rmse = np.sqrt(np.sum(error**2)/len(error))
rmse
#Calculating RMSE using Metrics library from Sklearn
print('Mean ABSOLUTE ERROR  =  ',metrics.mean_absolute_error(y_test,y_pred))
print('Mean SQUARED ERROR   =  ',metrics.mean_squared_error(y_test,y_pred))
print('Root Mean SQUARED ERROR   =  ',np.sqrt(metrics.mean_squared_error(y_test,y_pred)))
from scipy.stats import anderson
from scipy.stats import shapiro
from scipy.stats import kstest
anderson(sd['Salary'])
shapiro(sd['Salary'])
kstest(sd['Salary'],'norm')
import statsmodels.api as sm
model_sum = sm.OLS(y_train,X_train).fit()
model_sum.summary()
