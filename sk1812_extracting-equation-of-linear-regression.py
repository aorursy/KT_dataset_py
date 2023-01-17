import numpy as np

import pandas as pd

# import matplotlib.pyplot and seaborn for ploting the graphs

import matplotlib.pyplot as plt

import seaborn as sns

# Import train_test_split for spliting of data into training data and test data

from sklearn.model_selection import train_test_split

# import LinearRegression for fitting Simple Linear Regression to the Training set

from sklearn.linear_model import LinearRegression

# Importing metrics for the evaluation of the model

from sklearn.metrics import r2_score,mean_squared_error
# read the dataset using pandas

data = pd.read_csv('../input/salary-data-simple-linear-regression/Salary_Data.csv')
# This displays the top 5 rows of the data

data.head()
# Provides some information regarding the columns in the data

data.info()
# this describes the basic stat behind the dataset used 

data.describe()
plt.figure(figsize=(12,6))

sns.pairplot(data,x_vars=['YearsExperience'],y_vars=['Salary'],size=7,kind='scatter')

plt.xlabel('YearsExperience')

plt.ylabel('Salary')

plt.title('Salary Prediction')

plt.show()
X = data['YearsExperience']

X.head()
y = data['Salary']

y.head()
X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.7,random_state=100)
# Create new axis for x column

X_train = X_train[:,np.newaxis]

X_test = X_test[:,np.newaxis]
# Fitting the model

lr = LinearRegression()

lr.fit(X_train,y_train)
# Predicting the Salary for the Test values

y_pred = lr.predict(X_test)
plt.scatter(X_train, y_train, color = 'red')

plt.plot(X_train, lr.predict(X_train), color = 'blue')

plt.title('Salary vs Experience (Training set)')

plt.xlabel('Years of Experience')

plt.ylabel('Salary')

plt.show()
plt.scatter(X_test, y_test, color = 'red')

plt.plot(X_train, lr.predict(X_train), color = 'blue')

plt.title('Salary vs Experience (Test set)')

plt.xlabel('Years of Experience')

plt.ylabel('Salary')

plt.show()
c = [i for i in range (1,len(y_test)+1,1)]

plt.plot(c,y_test,color='r',linestyle='-')

plt.plot(c,y_pred,color='b',linestyle='-')

plt.xlabel('Salary')

plt.ylabel('index')

plt.title('Prediction')

plt.show()
c = [i for i in range(1,len(y_test)+1,1)]

plt.plot(c,y_test-y_pred,color='green',linestyle='-')

plt.xlabel('index')

plt.ylabel('Error')

plt.title('Error Value')

plt.show()
# calculate Mean square error

mse = mean_squared_error(y_test,y_pred)
# Calculate R squared value

rsq = r2_score(y_test,y_pred)
print('mean squared error :',mse)

print('r squared value :',rsq)
# Just plot actual and predicted values for more insights

plt.figure(figsize=(12,6))

plt.scatter(y_test,y_pred,color='r',linestyle='-')

plt.show()
# Intecept and coeff of the line

c=lr.intercept_

m=lr.coef_

print('Intercept of the model:',lr.intercept_)

print('Coefficient of the line:',lr.coef_)
# the exact equation of line

print("y =",m[0],"x+",c)