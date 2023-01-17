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
from sklearn.linear_model import LinearRegression #linear regression
from sklearn.model_selection import train_test_split # split train and test data 
from sklearn.preprocessing import PolynomialFeatures #polynomial regression
import matplotlib.pyplot as plt 
import seaborn as sns
df = pd.read_excel('/kaggle/input/car-consume/measurements2.xlsx')
df1 = pd.DataFrame(df, columns = ['distance', 'consume', 'temp_outside', 'speed', 'gas_type', 'rain']) # features we want to select
df1['gas_type'] = df1['gas_type'].map({'SP98': 1, 'E10': 0}) # mapping so that the letters are changed into numbers
y_data = df1['consume']
x_data = df1.drop('consume', axis=1)
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.15, random_state=1)
lm = LinearRegression()
lm.fit(x_train, y_train)
lm.score(x_test, y_test) # R^2 score for test set
lm.score(x_train, y_train) # R^2 score for training set
# residual plot 
width = 12
height = 10
plt.figure(figsize=(width, height))
sns.residplot(df1['distance'], df1['consume'])
plt.show()
pr = PolynomialFeatures(degree=2)
x_train_pr = pr.fit_transform(x_train)
x_test_pr = pr.fit_transform(x_test)
poly = LinearRegression()
poly.fit(x_train_pr, y_train)
# predicting with polynomial regression
Yhat_test_pr = poly.predict(x_test_pr)
poly.score(x_train_pr, y_train)

poly.score(x_test_pr, y_test)
def DistributionPlot(RedFunction, BlueFunction, RedName, BlueName, Title):
    width = 12
    height = 10
    plt.figure(figsize=(width, height))

    ax1 = sns.distplot(RedFunction, hist=False, color="r", label=RedName)
    ax2 = sns.distplot(BlueFunction, hist=False, color="b", label=BlueName, ax=ax1)

    plt.title(Title)
    plt.xlabel('Features')
    plt.ylabel('Prediction (cosumption)')

    plt.show()
    plt.close()
Title = 'Distribution  Plot of  Predicted Value Using Training Data vs Training Data Distribution'
DistributionPlot(y_test, Yhat_test_pr,"Actual Values (Test)", "Predicted Values (Test)", Title)
Yhat_train = lm.predict(x_train)
Yhat_test = lm.predict(x_test)
DistributionPlot(y_test, Yhat_test,"Actual Values (Test)", "Predicted Values (Test)", Title)
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, Yhat_test_pr)
print('The mean square error of price and predicted value for PR model is: ', mse)
mse = mean_squared_error(y_test, Yhat_test)
print('The mean square error of price and predicted value for LR model isis: ', mse)