# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as s
s.set()

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
data = pd.read_csv('/kaggle/input/advertising-dataset/advertising.csv')  
print(data.head())

data.info()
data.describe()

# data cleaning 

data.isnull().sum()

# no null value data is clean

# outlier analysis using boxplot

f, axs = plt.subplots(1,3, figsize=(15,5))

plt1=s.boxplot(x=data['TV'], ax=axs[0])
plt2=s.boxplot(x=data['Radio'], ax=axs[1])
plt3=s.boxplot(x=data['Newspaper'], ax=axs[2])
plt.show()

# target variable

s.boxplot(x=data['Sales'])

#how sales are related with other variables using scatter plot

s.pairplot(data, x_vars=['TV', 'Radio', 'Newspaper'], y_vars='Sales', height=4, kind='scatter')
plt.show()

#correlation using heatmap

#s.heatmap(data.corr(), cmap='YlGnBu', annot=True)

# using heatmap found TV is highly related to Sales prediction, lets go with linear regression model

x=data['TV']
y=data['Sales']

# train test split

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8,test_size=0.2,random_state=10)

#building a linear model using statsmodel

import statsmodels.api as sm

x_train_sm = sm.add_constant(x_train)

# fit regression line using OLS

results = sm.OLS(y_train, x_train_sm).fit()
print(results.summary())

yhat=6.948 + 0.054*x_train
plt.scatter(x_train, y_train)
plt.plot(x_train, yhat, 'r')
plt.xlabel('TV')
plt.ylabel('Sales')
plt.show()


# predictions on the test set

# Add a constant to X_test
x_test_sm = sm.add_constant(x_test)

# Predict the y values corresponding to X_test_sm
y_pred = results.predict(x_test_sm)

y_pred.head()

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

np.sqrt(mean_squared_error(y_test, y_pred))

r_square = r2_score(y_test, y_pred)
print(r_square)


#visualize test data 

yhat=6.948 + 0.054*x_test
plt.scatter(x_test, y_test)
plt.plot(x_test, yhat, 'r')
plt.xlabel('TV')
plt.ylabel('Sales')
plt.show()



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
