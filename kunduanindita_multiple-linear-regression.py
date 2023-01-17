#importing the libraries

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns
#importing the dataset

companies = pd.read_csv('../input/startup50/50_Startups.csv')



#extracting the independent and dependent variables

X = companies.iloc[:, :-2].values

y = companies.iloc[:, -1].values



companies.head()
X
y
# data visualizaton

# building the correlation matrix

sns.heatmap(companies.corr())
print(X)
# splitting

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
# fitting multiple linear regression to the training set

from sklearn.linear_model import LinearRegression

regressor =  LinearRegression()

regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

y_pred
print(regressor.coef_)
print(regressor.intercept_)
from sklearn.metrics import r2_score

r2Score = r2_score(y_test, y_pred)



print(r2Score)