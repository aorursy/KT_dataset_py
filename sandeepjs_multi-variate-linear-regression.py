import os

import pandas as pd

from scipy import stats

import numpy as np

import matplotlib.pyplot as plt

import seaborn as seabornInstance

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn import metrics

%matplotlib inline
data = pd.read_csv('../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')
data.shape
data.describe() #view statistical details of the data
#Download EDA report of the data

import pandas_profiling

import seaborn

eda_report = pandas_profiling.ProfileReport(data)

eda_report.to_file("winequality_eda.html")
data.isnull().any() #Lets check for null values
X = data[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates','alcohol']].values

y = data['quality'].values
plt.figure(figsize=(15,10))

plt.tight_layout()

seabornInstance.distplot(data['quality'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
lm = LinearRegression()

lm.fit(X_train,y_train)
y_pred = lm.predict(X_test)
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df1 = df.head(25)

df1
df1.plot(kind='bar',figsize=(10,8))

plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')

plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')

plt.show()
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  

print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))