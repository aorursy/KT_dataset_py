import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

%matplotlib inline
df = pd.read_csv('../input/position-salaries/Position_Salaries.csv')
df.head()
df.info()
df.describe()
# Step 1 - Load Data



X = df.iloc[:, 1:2].values

y = df.iloc[:, 2].values
# Step 2 - Fitting Linear Regression



from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()

lin_reg.fit(X, y)
# Step 3 - Visualize Linear Regression Results



plt.scatter(X, y, color="red")

plt.plot(X, lin_reg.predict(X))

plt.title("Linear Regression")

plt.xlabel("Level")

plt.ylabel("Salary")

plt.show()
# Step 4 Linear Regression prediction



new_salary_pred = lin_reg.predict([[6.5]])

print('The predicted salary of a person at 6.5 Level is ',new_salary_pred)
# Step 5 - Convert X to polynomial format



from sklearn.preprocessing import PolynomialFeatures

poly_reg_2 = PolynomialFeatures(degree=2)

X_poly_2 = poly_reg_2.fit_transform(X)
# Step 6 - Passing X_poly to LinearRegression



lin_reg_2 = LinearRegression()

lin_reg_2.fit(X_poly_2, y)
# Step 7 - Visualize Poly Regression Results



plt.scatter(X, y, color="red")

plt.plot(X, lin_reg_2.predict(X_poly_2))

plt.title("Poly Regression - Degree 2")

plt.xlabel("Level")

plt.ylabel("Salary")

plt.show()
# Step 8 Polynomial Regression prediction



new_salary_pred = lin_reg_2.predict(poly_reg_2.fit_transform([[6.5]]))

print('The predicted salary of a person at 6.5 Level is ',new_salary_pred)
# Step 9 - Change degree to 3 and run steps 5-8



# Step 5 - Convert X to polynomial format

from sklearn.preprocessing import PolynomialFeatures

poly_reg_3 = PolynomialFeatures(degree=3)

X_poly_3 = poly_reg_3.fit_transform(X)



 # Step 6 - Passing X_poly to LinearRegression

lin_reg_3 = LinearRegression()

lin_reg_3.fit(X_poly_3, y)



# Step 7 - Visualize Poly Regression Results

plt.scatter(X, y, color="red")

plt.plot(X, lin_reg_3.predict(X_poly_3))

plt.title("Poly Regression Degree 3")

plt.xlabel("Level")

plt.ylabel("Salary")

plt.show()



# Step 8 Polynomial Regression prediction

new_salary_pred = lin_reg_3.predict(poly_reg_3.fit_transform([[6.5]]))

print('The predicted salary of a person at 6.5 Level is ',new_salary_pred)
# Step 10 - Change degree to 4 and run steps 5-8



# Step 5 - Convert X to polynomial format

from sklearn.preprocessing import PolynomialFeatures

poly_reg_4 = PolynomialFeatures(degree=4)

X_poly_4 = poly_reg_4.fit_transform(X)



 # Step 6 - Passing X_poly to LinearRegression

lin_reg_4 = LinearRegression()

lin_reg_4.fit(X_poly_4, y)



# Step 7 - Visualize Poly Regression Results

plt.scatter(X, y, color="red")

plt.plot(X, lin_reg_4.predict(X_poly_4))

plt.title("Poly Regression Degree 3")

plt.xlabel("Level")

plt.ylabel("Salary")

plt.show()



# Step 8 Polynomial Regression prediction

new_salary_pred = lin_reg_4.predict(poly_reg_4.fit_transform([[6.5]]))

print('The predicted salary of a person at 6.5 Level is ',new_salary_pred)