import numpy as np

import pandas as pd

import matplotlib as mpl

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

dataFrame = pd.read_csv('../input/boston-house-prices/housing.csv', header=None, delimiter=r"\s+", names=column_names)
dataFrame.shape
dataFrame.head(5)
#Seaborn Plot

plt.figure(figsize=(16,16))

ax = sns.heatmap(dataFrame.corr(), linewidth=0.5, vmin=-1,

cmap='coolwarm', annot=True)

plt.title('Correlation heatmap')

plt.show()
g = sns.jointplot("DIS", "AGE", data=dataFrame, kind="hex")
g = sns.pairplot(dataFrame[['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']], palette = 'hls',size=2)

g.set(xticklabels=[]);
dataFrame.info
dataFrame.dtypes
dataFrame.isnull().any()
dataFrame.describe()
X = dataFrame.drop(['MEDV'], axis = 1)

y = dataFrame['MEDV']
# Finding out the correlation between the features

corr = dataFrame.corr()

corr.shape
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=231)
X_train.head(5)
X_test.head(5)
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_train, y_train)
y_predict = regressor.predict(X_test)
y_predict
y_test
#Fitting the regular linear regression model to the training data set

import statsmodels.api as sm



X_train_sm = X

X_train_sm = sm.add_constant(X_train_sm)



linearRegressionModel_sm = sm.OLS(y,X_train_sm.astype(float)).fit()
print(linearRegressionModel_sm.summary())
linearRegressionModel_sm.summary()