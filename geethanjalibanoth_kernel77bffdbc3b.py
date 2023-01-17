# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels
import statsmodels.api as sm
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Importing Dataset
data = pd.read_csv('/kaggle/input/salary-data-simple-linear-regression/Salary_Data.csv')
data.head()

#Understanding the Structure of the Data
data.shape
#To check any Missing Values
data.info()
#Summary of Important Statistics
data.describe()
#visualize the data
sns.regplot(x='YearsExperience',y='Salary',data=data)
#There is a High positive correlation between salary and YearsExperience
sns.heatmap(data.corr(),annot=True)
#create X and y
X=data['YearsExperience']
y=data['Salary']
#create train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7, test_size = 0.3, random_state = 100)

#Training the Model
X_train_sm=sm.add_constant(X_train)
#Fitting the Model
lr=sm.OLS(y_train,X_train_sm)
lr_model=lr.fit()
lr_model.params
lr_model.summary()
#Salary=25202.887786+9731.203838*YearsExperience
#p-value is 0 so both of the coefficients are significant
#R-Squared value is 0.949 which is pretty high
#If the 'Prob (F-statistic)' is less than 0.05, we can conclude that the overall model fit is significant.
#Evaluating the model
y_train_pred=lr_model.predict(X_train_sm)
plt.scatter(X_train,y_train)
plt.plot(X_train,y_train_pred,'r')
plt.show()

#Residual Analysis
res=y_train-y_train_pred
res
#Plotting the Residuals
plt.figure()
sns.distplot(res)
plt.title("Residual Plot")
#check for any patterns
plt.scatter(X_train,res)
plt.show()
#Predictions and evaluation on test set
X_test_sm=sm.add_constant(X_test)
y_test_pred=lr_model.predict(X_test_sm)
y_test_pred
#Evaluate the model,r-squared
r2=r2_score(y_true=y_test,y_pred=y_test_pred)
r2
#r2 on train
r2=r2_score(y_true=y_train,y_pred=y_train_pred)
r2
#mean-squared-error
mean_squared_error(y_true=y_test,y_pred=y_test_pred)