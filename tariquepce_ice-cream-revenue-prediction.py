import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
IceCream = pd.read_csv("/kaggle/input/ice-cream-revenue/IceCreamData.csv")

IceCream.head(100)
IceCream.tail()
IceCream.describe()
IceCream.info()
sns.jointplot(x='Temperature', y='Revenue', data = IceCream, color = 'gray')
sns.pairplot(IceCream)
sns.lmplot(x='Temperature', y='Revenue', data=IceCream)
y = IceCream['Revenue']
X = IceCream[['Temperature']]
X
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
X_train.shape
from sklearn.linear_model import LinearRegression
regressor = LinearRegression(fit_intercept = True)
regressor.fit(X_train,y_train)
print('Linear Model Coefficient (m): ', regressor.coef_)
print('Linear Model Coefficient (b): ', regressor.intercept_)
y_predict = regressor.predict( X_test)
y_predict
y_test
plt.scatter(X_train, y_train, color = 'gray')
plt.plot(X_train, regressor.predict(X_train), color = 'red')
plt.ylabel('Revenue [dollars]')
plt.xlabel('Temperature [degC]')
plt.title('Revenue Generated vs. Temperature @Ice Cream Stand(Training dataset)')
# VISUALIZE TEST SET RESULTS
plt.scatter(X_test, y_test, color = 'gray')
plt.plot(X_test, regressor.predict(X_test), color = 'red')
plt.ylabel('Revenue [dollars]')
plt.xlabel('Hours')
plt.title('Revenue Generated vs. Hours @Ice Cream Stand(Test dataset)')