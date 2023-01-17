import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn import datasets
seed = 0
np.random.seed(seed)
from sklearn.datasets import load_boston
# Load the Boston Housing dataset from sklearn
boston = load_boston()
bos = pd.DataFrame(boston.data)
# give our dataframe the appropriate feature names
bos.columns = boston.feature_names
# Add the target variable to the dataframe
bos['Price'] = boston.target
# For student reference, the descriptions of the features in the Boston housing data set
# are listed below
boston.DESCR
bos.head()
# Select target (y) and features (X)
X = bos.iloc[:,:-1]
y = bos.iloc[:,-1]
# Split the data into a train test split
x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.25, random_state=seed, shuffle=True)
lreg = LinearRegression()
lreg.fit(x_train,y_train)
pred = lreg.predict(x_test)
rmse = np.sqrt(np.mean((y_test-pred)**2))

print("Root Mean Squared Error: {}".format(rmse))
print("R^2: {}".format(lreg.score(x_test, y_test)))
print("Adjusted R^2: {}".format(1 - (1-lreg.score(x_test, y_test))*(len(y_test)-1)/(len(y_test)-x_test.shape[1]-1)))
# Fit a linear regression model using Ridge
ridge = Ridge()
ridge.fit(x_train,y_train) 
pred = lreg.predict(x_test)
rmse = np.sqrt(np.mean((y_test-pred)**2))

print("Root Mean Squared Error: {}".format(rmse))
print("R^2: {}".format(lreg.score(x_test, y_test)))
print("Adjusted R^2: {}".format(1 - (1-lreg.score(x_test, y_test))*(len(y_test)-1)/(len(y_test)-x_test.shape[1]-1)))
# Fit a linear regression model using lasso
lasso= Lasso()
lasso.fit(x_train,y_train) 
pred = lreg.predict(x_test)
rmse = np.sqrt(np.mean((y_test-pred)**2))

print("Root Mean Squared Error: {}".format(rmse))
print("R^2: {}".format(lreg.score(x_test, y_test)))
print("Adjusted R^2: {}".format(1 - (1-lreg.score(x_test, y_test))*(len(y_test)-1)/(len(y_test)-x_test.shape[1]-1)))
#Remove dependent Variable 
X = bos[['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD','PTRATIO', 'B', 'LSTAT']]
# Split the data into a train test split
x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.25, random_state=seed, shuffle=True)
lreg = LinearRegression()
lreg.fit(x_train,y_train)
pred = lreg.predict(x_test)
rmse = np.sqrt(np.mean((y_test-pred)**2))

print("Root Mean Squared Error: {}".format(rmse))
print("R^2: {}".format(lreg.score(x_test, y_test)))
print("Adjusted R^2: {}".format(1 - (1-lreg.score(x_test, y_test))*(len(y_test)-1)/(len(y_test)-x_test.shape[1]-1)))
#Remove dependent Variable 
#X = bos[['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD','PTRATIO', 'B', 'LSTAT']]
X = bos[['CRIM','ZN', 'INDUS','CHAS', 'NOX','RM', 'DIS','RAD','PTRATIO', 'B','LSTAT']]
# Split the data into a train test split
x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.25, random_state=seed, shuffle=True)
lreg = LinearRegression()
lreg.fit(x_train,y_train)
pred = lreg.predict(x_test)
rmse = np.sqrt(np.mean((y_test-pred)**2))

print("Root Mean Squared Error: {}".format(rmse))
print("R^2: {}".format(lreg.score(x_test, y_test)))
print("Adjusted R^2: {}".format(1 - (1-lreg.score(x_test, y_test))*(len(y_test)-1)/(len(y_test)-x_test.shape[1]-1)))
#Remove dependent Variable 
#X = bos[['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD','PTRATIO', 'B', 'LSTAT']]
X = bos[['CRIM','ZN', 'INDUS','CHAS', 'NOX','RM', 'DIS','RAD','TAX','AGE','PTRATIO', 'B','LSTAT']]
# Split the data into a train test split
x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.25, random_state=seed, shuffle=True)
lreg = LinearRegression()
lreg.fit(x_train,y_train)
pred = lreg.predict(x_test)
rmse = np.sqrt(np.mean((y_test-pred)**2))
print("Root Mean Squared Error: {}".format(rmse))
print("R^2: {}".format(lreg.score(x_test, y_test)))
print("Adjusted R^2: {}".format(1 - (1-lreg.score(x_test, y_test))*(len(y_test)-1)/(len(y_test)-x_test.shape[1]-1)))
bos_corr=pd.DataFrame(bos).corr()
bos_corr.style.background_gradient(cmap='coolwarm')

plt.scatter(y_test,pred)
plt.xlabel('Price')
plt.ylabel('Predicted_Price')
plt.show()