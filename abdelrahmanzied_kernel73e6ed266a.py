# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Import
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error
import matplotlib.pyplot as plt
#Data
data = pd.read_csv('/kaggle/input/salary-data-simple-linear-regression/Salary_Data.csv')
X = data.iloc[:, :1]
y = data.iloc[:, 1:]
data.head()
#Splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33, random_state=0, shuffle=True)
#Model
LinearRegressionModel = LinearRegression(copy_X=True, fit_intercept=True, normalize=True)
LinearRegressionModel.fit(X,y)
print("Linear Regression Train Score: ",LinearRegressionModel.score(X_train, y_train))
print("Linear Regression Test Score: ",LinearRegressionModel.score(X_test, y_test))
#Predict
y_pred = LinearRegressionModel.predict(X_test)
#Metrices
MAEValue = mean_absolute_error(y_test, y_pred, multioutput='uniform_average')
print('Mean Absolute Error: ', MAEValue)

MSEValue = mean_squared_error(y_test, y_pred, multioutput='uniform_average')
print('Mean Squared Error: ', MSEValue)

MdSEValue = median_absolute_error(y_test, y_pred, multioutput='uniform_average')
print('Median Absolute Error: ', MdSEValue)
#Graph
plt.style.use('seaborn-whitegrid')
ax = plt.axes()
ax.set(xlabel='X-Train', ylabel='Y-Train and Y-Predict', title='Relationship between Y-Test and Y-Predict')
plt.scatter(X_test,y_test, alpha=.5, label='Test')
plt.plot(X_test,y_pred, label='Predict')
plt.legend(prop={'size': 16})