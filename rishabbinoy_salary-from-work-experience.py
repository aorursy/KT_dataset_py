from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
emp_data = pd.read_csv('Salary_Data.csv')
print(emp_data.head()) # Prints the first few entries of the dataset
print(emp_data.describe()) # Prints some statistical properties of the dataset
X = emp_data.YearsExperience
y = emp_data.Salary
X_train, X_test, y_train, y_test = train_test_split(X , y, train_size = 0.8, test_size = 0.2, random_state = 1)
my_model = LinearRegression()
X_train = X_train.values.reshape(-1,1) # Since model.fit() expects a 2D array
X_test = X_test.values.reshape(-1,1)
my_model.fit(X_train, y_train)
print("{:.2f}x + {:.2f}".format(my_model.coef_[0],my_model.intercept_)) 
pred = my_model.predict(X_test)
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test, pred)
print('Mean absolute error is %0.2f'%mae)
print('Model Score: %0.2f/1.0'%my_model.score(X_test,y_test))
import numpy as np
example = np.array([float(input())]) # For testing a single input
example = example.reshape(1,-1)
pred_example = my_model.predict(example)
print('$%0.2f'%pred_example)
