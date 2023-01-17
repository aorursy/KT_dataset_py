import numpy as np

import matplotlib.pyplot as plt

import pandas as pd
# Importing the dataset

dataset = pd.read_csv('../input/50_Startups.csv')

# features = ['R&D Spend', 'Administration', 'Marketing Spend', 'State']

X = dataset.iloc[:, :-1].values

# X = dataset[features]

y = dataset.iloc[:, -1].values
# Encoding categorical data

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder = LabelEncoder()

X[:, 3] = labelencoder.fit_transform(X[:, 3])

onehotencoder = OneHotEncoder(categorical_features = [3])

X = onehotencoder.fit_transform(X).toarray()
#Avoiding the dummy variable trap

X = X[:, 1:]
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

X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis=1)
X_opt = X[:, [0, 3, 5]]

reg_ols = sm.OLS(endog = y, exog = X_opt).fit()

reg_ols.summary()