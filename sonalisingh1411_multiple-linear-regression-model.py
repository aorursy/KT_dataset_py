# Importing the libraries

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

dataset = pd.read_csv("../input/50_Startups.csv")
#Lets check the dataset!

dataset.head()
dataset.isnull().sum()
dataset["State"].unique()
#Plot R&D vs Profit............

x1 = dataset.iloc[:, 0].values

y1 = dataset.iloc[:, -1].values

plt.scatter(x1,y1,color='Green',s=50)

plt.xlabel('R&D')

plt.ylabel('Profit')

plt.title('R&D vs Profit')

plt.show()
#Plot Administration vs Profit

x1 = dataset.iloc[:, 1].values

y1 = dataset.iloc[:, -1].values

plt.scatter(x1,y1,color='Red',s=50)

plt.xlabel('Administration')

plt.ylabel('Profit')

plt.title('Administration vs Profit')

plt.show()
#Plot Marketing Spend vs Profit

x1 = dataset.iloc[:, 2].values

y1 = dataset.iloc[:, -1].values

plt.scatter(x1,y1,color='Black',s=50)

plt.xlabel('Marketing Spend')

plt.ylabel('Profit')

plt.title('Marketing Spend vs Profit')

plt.show()
#Plot State vs Profit

x1 = dataset.iloc[:, 3].values

y1 = dataset.iloc[:, -1].values

plt.scatter(x1,y1,color='Blue',s=50)

plt.xlabel('State')

plt.ylabel('Profit')

plt.title('State vs Profit')

plt.show()
X = dataset.iloc[:, :-1].values

print(X)

y = dataset.iloc[:, 4].values

print(y)


# Encoding categorical data

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder = LabelEncoder()

X[:, 3] = labelencoder.fit_transform(X[:, 3])

onehotencoder = OneHotEncoder(categorical_features = [3])

X = onehotencoder.fit_transform(X).toarray()


# Avoiding the Dummy Variable Trap

X = X[:, 1:]

print(X)
# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting Multiple Linear Regression to the Training set

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_train, y_train)



# Predicting the Test set results

y_pred = regressor.predict(X_test)
import statsmodels.formula.api as sm

X

X_opt = X[:,[0,1,2,3,4]]

X_opt

regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()

regressor_OLS.summary()
