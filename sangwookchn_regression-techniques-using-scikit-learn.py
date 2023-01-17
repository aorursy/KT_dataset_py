#data preprocessing

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('../input/salary-data/Salary_Data.csv')
x = data.iloc[:, :-1].values
y = data.iloc[:, 1].values

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 1/3, random_state = 42)

#No need to do feature scaling, as the library automatically takes care of this.

#fitting regressor
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)


#predicting the test set results
y_pred = regressor.predict(X_test) #vector of all predictions of the dependent variable
y_pred
#visualize the training set results
plt.scatter(X_train, Y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title("years vs salary")
plt.xlabel("number of years")
plt.ylabel("salary (dollars)")
plt.show()
#visualizing test set results
plt.scatter(X_test, Y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue') #this should not change as the regressor fitted to train set should be shown
plt.title("years vs salary")
plt.xlabel("number of years")
plt.ylabel("salary (dollars)")
plt.show()
data2 = pd.read_csv('../input/m-50-startups/50_Startups.csv')
x = data2.iloc[:, :-1]
y = data2.iloc[:, 4]

from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical

xtemp = x.iloc[:, 3]
labelencoder = LabelEncoder()
xtemp = labelencoder.fit_transform(xtemp)
xtemp = pd.DataFrame(to_categorical(xtemp))

x = x.drop(['State'], axis = 1) #In pandas axis = 1 --> column
x = pd.concat([x, xtemp], axis = 1)

# x = x.iloc[:, :-1] This is to avoid dummay variable trap. However, the library already takes care of this, so no need.

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)


#Linear Regression

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

y_pred = regressor.predict(X_test)
y_pred, Y_test

print(x.shape[1])
#Find an optimal team of independent variables, so that each variable has significant impact on the prediction. 
# --> Backward elimination

import statsmodels.formula.api as sm
x = np.array(x) #use numpy arrays instead of DataFrames for more useful functions. DataFrames are useful for preparing dataset
x = np.append(np.ones((x.shape[0], 1), dtype = 'int'), x, axis = 1) #(x.shape[0], 1).astype(int) does not work
#Above is done to add constant to the model, which is necessary for Ordinary Least Squares to work
x_opt = x[:, [0,1,2,3,4,5]] #needs to specify all the indexes, so that individual index is evaluated.
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary() #shows statistical summary
'''The significance value is set to 0.05 If P-value is lower than this, 
then it is significant. If higher, it is less significant. Therefore, variables
with higher P_values need to be removed, as they do not have large impact. 
This is called backward elimination. 

In this case, as x4 has 0.990, it needs to be removed'''

x_opt = x[:, [0,1,2,3,4,5]] #needs to specify all the indexes, so that individual index is evaluated.
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary() #shows statistical summary




#repeat the step --> remove the insignificant variable, fit it, repeat it.

x_opt = x[:, [0,1,3,5]] #needs to specify all the indexes, so that individual index is evaluated.
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary() #shows statistical summary
x_opt = x[:, [0,1,3]] #needs to specify all the indexes, so that individual index is evaluated.
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary() #shows statistical summary
#To check if the team of variables are correct
x_show = pd.DataFrame(x)
# x_show

#Therefore, only using these variables 
data3 = pd.read_csv('../input/polynomial-position-salary-data/Position_Salaries.csv')
x = data3.iloc[:, 1:2].values #1:2 is done instead of only 1, because independent variable should be a matrix.
y = data3.iloc[:, 2].values

# X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
# Use whole dataset to train

data3
plt.scatter(data3.iloc[:, 1], y)
plt.title('Level vs Salary')
plt.xlabel("Level")
plt.ylabel("Salary (dollrs)")
plt.show()
from sklearn.preprocessing import PolynomialFeatures
#transforms the x matrix into a new matrix that has x1, x2, x3 --- columns
poly_reg = PolynomialFeatures(degree = 2) #specify the degree -> how many terms
x_poly = poly_reg.fit_transform(x)
x_poly
lin_reg = LinearRegression()
lin_reg.fit(x_poly, y)

y_pred = lin_reg.predict(poly_reg.fit_transform(x)) #this is used instead of x_poly, so that this model will work for any matrix input x 

plt.figure(2)
plt.scatter(x, y)
plt.plot(x, y_pred)
plt.show()
#improving the model --> add degrees to make it more complex

poly_reg = PolynomialFeatures(degree = 4) #specify the degree -> how many terms
x_poly = poly_reg.fit_transform(x)
x_poly

lin_reg = LinearRegression()
lin_reg.fit(x_poly, y)

y_pred = lin_reg.predict(poly_reg.fit_transform(x)) #this is used instead of x_poly, so that this model will work for any matrix input x 

#This is to get a more continuous curve, by plotting more x values.
x_grid = np.arange(min(x), max(x), 0.1) #0.1 --> increment 
x_grid = x_grid.reshape(x_grid.shape[0], 1)

plt.figure(2)
plt.scatter(x, y)
plt.plot(x_grid, lin_reg.predict(poly_reg.fit_transform(x_grid)))
plt.show()
data3 = pd.read_csv('../input/polynomial-position-salary-data/Position_Salaries.csv')
x = data3.iloc[:, 1:2].values #1:2 is done instead of only 1, because independent variable should be a matrix.
y = data3.iloc[:, 2].values
y = y.reshape(y.shape[0], 1)
x = x.reshape(x.shape[0], 1)

print(y.shape)
# X_train, X_test, Y_train, Y_test = train_test_split()

#svr does not have feature scaling built in
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
x = sc_x.fit_transform(x)
y = sc_y.fit_transform(y)

from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(x, y)

y_pred = sc_y.inverse_transform(regressor.predict(sc_x.transform(np.array([[6.5]])))) #input scaled value, then inverse scale the predicted value
y_pred
data3 = pd.read_csv('../input/polynomial-position-salary-data/Position_Salaries.csv')
x = data3.iloc[:, 1:2].values #1:2 is done instead of only 1, because independent variable should be a matrix.
y = data3.iloc[:, 2].values

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(x, y)

y_pred = regressor.predict(np.array([[6.5]]))

x_grid = np.arange(min(x), max(x), 0.01)
x_grid = x_grid.reshape(len(x_grid), 1)
plt.figure(3)
plt.plot(x_grid, regressor.predict(x_grid))
plt.show()

#Notice how average is used to represent each interval.
data3 = pd.read_csv('../input/polynomial-position-salary-data/Position_Salaries.csv')
x = data3.iloc[:, 1:2].values #1:2 is done instead of only 1, because independent variable should be a matrix.
y = data3.iloc[:, 2].values

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 5000, criterion = 'mse', random_state = 0) #can tune the n_estimators
regressor.fit(x, y)

y_pred = regressor.predict([[6.5]])

x_grid = np.arange(min(x), max(x), 0.01)
x_grid = x_grid.reshape(len(x_grid), 1)
plt.plot(x_grid, regressor.predict(x_grid))
plt.show()
print(y_pred)
