import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
Data=pd.read_csv(r'../input/percentage-marks-vs-hours-studied/Task1.csv')
Data
Data.shape
Data.head()
Data.describe()
Data.plot(x="Hours",y="Scores",style="o")

plt.title("Hours Vs Percentage")

plt.xlabel("Hours")

plt.ylabel("Percentage")

plt.show()
X = Data.iloc[:, :-1].values

y = Data.iloc[:, 1].values
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_train, y_train)
print(regressor.intercept_)
print(regressor.coef_)
y_pred = regressor.predict(X_test)
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

df
from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))

print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))