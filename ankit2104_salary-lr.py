import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

salary_set = pd.read_csv('../input/salary/Salary.csv')
salary_set.head()
#To check for any null values
salary_set.isnull().sum()
#Check the Data Types
salary_set.dtypes
#Rename a column if any wrong name using python dictionaries(cleaning data)
new_cl_name = {'YearsExperience' : 'yoe'}
salary_set = salary_set.rename(columns = new_cl_name)
salary_set.head()
#Assigning the dependent and independent values
x = salary_set['yoe']
y = salary_set['Salary']

#Exploring the data
plt.scatter(x,y,color = 'red')
plt.title('YOE vs Salary')
plt.xlabel('Years Of Experience')
plt.ylabel('Salary')
#Measure the corelation/strength between 2 variables

salary_set.corr()

#Very Strong Relationship(|r|>0.8)
#Strong relationship(0.6=<|r|)
#Moderate relationship(0.4=<|r|)
#Weak relationship(0.2=<|r|)
#Statistical Summary
salary_set.describe()
#Training and splitting the data
X = salary_set.yoe
Y = salary_set['Salary']

Xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size = 1/3, random_state = 1)
Xtrain = Xtrain.values.reshape(-1,1)# Since model.fit() expects a 2D array
xtest = xtest.values.reshape(-1,1)
Xtrain.shape
#ytrain.shape 
#xtest.shape  
#ytest.shape  8
#create a regression model
regression_model = LinearRegression()

#pass the Xtrain & Ytrain dataset
regression_model.fit(Xtrain, ytrain) 
#find the intercept and coefficient
#intercept = regression_model.intercept_
#coefficient = regression_model.coef_
print("{:.2f}x + {:.2f}".format(regression_model.coef_[0],regression_model.intercept_))
#Predicting the model
y_predict = regression_model.predict(xtest)
y_predict[:5]
#Acurracy using r2 method
model_r2 = r2_score(ytest,y_predict)
print(model_r2*100)
plt.scatter(X,Y,color = 'red')
plt.plot(xtest,y_predict)
plt.show()
