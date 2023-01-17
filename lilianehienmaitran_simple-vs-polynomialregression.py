#import all the libraries:
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#upload the file and identify independent and dependent features
dataset = pd.read_csv('/kaggle/input/years-of-experience-and-salary-dataset/Salary_Data.csv')
X = dataset.iloc[:, 0].values
Y = dataset.iloc[:, 1].values
#quickly check the data to make sure it's correct 
dataset.head()
# We will change X from a line to a column
X = X.reshape(len(X), -1)
#Using Linear Regression to train the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, Y)
#Let's see what is the coefficient and the intercept
print(lin_reg.coef_)
print(lin_reg.intercept_)
#Let's see if person with 3.5 years of experience, how much the salary this person will make, according to this model:
estimate_salary = lin_reg.predict([[3.5]])
np.set_printoptions(precision=2)
print(estimate_salary)
#how's about 6.5 year of experience? 
estimate_salary = lin_reg.predict([[6.5]])
np.set_printoptions(precision=2)
print(estimate_salary)
#Let's push it up to 9.5 year of experience:
estimate_salary = lin_reg.predict([[9.5]])
np.set_printoptions(precision=2)
print(estimate_salary)
#Let's see how Simple Linear Regression look like: 
plt.scatter(X, Y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('year of experience & salary-Simple Linear Regression')
plt.xlabel('Year of Experience')
plt.ylabel('Salary')
plt.show()
#Let's train the dataset using Polynomial Regression: 
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, Y)

#Let asks this new model the same questions that we asked the previous model:
#with 3.5 years of experience, the salary is: 
Y_pred = lin_reg_2.predict(poly_reg.fit_transform([[3.5]]))
print(Y_pred)

#6.5 years of experience, the salary is: 
Y_pred = lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
print(Y_pred)

#how's about at 9.5 years of experience?
Y_pred = lin_reg_2.predict(poly_reg.fit_transform([[9.5]]))
print(Y_pred)
#Let's see how Polynominal Regression look like: 
plt.scatter(X, Y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('year-salary-Simple Linear Regression')
plt.xlabel('Year of experience')
plt.ylabel('Salary')
plt.show()