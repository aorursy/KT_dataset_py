import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset = pd.read_csv('../input/position-salaries/Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values # level in dataset
y = dataset.iloc[:, -1].values # salary in dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y) # linear reg model fitted on level and salary
X
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
# degree meaning to the power if 4 then there will be 5 columns column 1 is all ones
# column 2 is of same number X
# column 3 is of number X^2(X to the power 2) i.e. X.X
# column 4 is of number X^3(X to the power 3) i.e. X.X.X
# column 5 is of number X^4(X to the power 4) i.e. X.X.X.X
# here we are giving the features which are in the polynomial increasing order 
# But the model is still the linear Regression so some call it as polynomial linear regression
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)
X_poly 
# u can see the array with X^0, X^1, X^2, X^3, X^4
# 2.401e+03 it means (2.401 * 10^3)
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')

'''
 poly_reg.fit_transform - it gives us a numpy array of 5 columns in a observation.
i.e X^0, X^1, X^2, X^3, X^4
lin_reg_2.predict - it takes the values mentioned above and predicts a single value, salary.
once we have defined a model and used to predict some values out of it,
we need to give features of same order i.e lin_reg_2.predict takes 
values which have 5 columns and it should be a numpy array.
if we give any data which doesn't have 5 columns or less than or more than the model will 
give u an error because the model is trained on the specific data.

'''
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
lin_reg.predict([[6.5]])

# simple linear regression model takes only the 2 dimensional array
lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))

# polynomial reg model takes the polynomial features only to predict.
print(lin_reg_2.coef_)

print(lin_reg.coef_)