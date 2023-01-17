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
import warnings

warnings.filterwarnings('ignore')



# Import the numpy and pandas package



import numpy as np

import pandas as pd



# Data Visualisation

import matplotlib.pyplot as plt 

import seaborn as sns
advertising = pd.DataFrame(pd.read_csv("/kaggle/input/advertising-dataset/advertising.csv"))



advertising.head()
advertising.shape
advertising.info()
advertising.describe()
advertising.isnull().sum()
# Let's see how Sales are related with other variables using scatter plot.

sns.pairplot(advertising, x_vars=['TV', 'Newspaper', 'Radio'], y_vars='Sales', height=4, aspect=1, kind='scatter')

plt.show()
# Let's see the correlation between different variables.

sns.heatmap(advertising.corr(), cmap="YlGnBu", annot = True)

plt.show()
X = advertising['TV']

y = advertising['Sales']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y,

         test_size = 0.3, random_state = 100)
print(X_train.shape)

print(X_test.shape)
X_train = X_train.values.reshape(-1,1)

X_test = X_test.values.reshape(-1,1)
print(X_train.shape)

print(X_test.shape)
print(X_test.shape)

print(y_test.shape)
from sklearn.linear_model import LinearRegression

reg = LinearRegression()





reg.fit(X_train,y_train)  # Fit  --> start learning 

reg.score(X_train,y_train)  # score --- how good is the model  ?
reg.coef_
reg.intercept_
plt.scatter(X_train, y_train)

plt.plot(X_train, 6.948 + 0.054*X_train, 'r')

plt.show()



# Best fit Line 
y_train_pred = reg.predict(X_train)

print(y_train_pred[0:5])

print(y_train[0:5])

res = (y_train - y_train_pred)

print(res[0:5])  # error  - difference between true value and predicted value 
fig = plt.figure()

sns.distplot(res, bins = 15)

fig.suptitle('Error Terms', fontsize = 15)                  # Plot heading 

plt.xlabel('y_train - y_train_pred', fontsize = 15)         # X-label

plt.show()
y_pred = reg.predict(X_test)



print(y_test[0:5])

print(y_pred[0:5])
from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score



#MSE =( (y_test_1 - y_pred_1 )^2 + ..... +   (y_test_60 - y_pred_60 )^2 )/60

#RMSE  = Square Root of MSE
#Returns the mean squared error; we'll take a square root

np.sqrt(mean_squared_error(y_test, y_pred))
r_squared = r2_score(y_test, y_pred)

r_squared



# R-Square value represent - how much variability is explained by model 
plt.scatter(X_test, y_test)

plt.plot(X_test, 6.948 + 0.054 * X_test, 'r')

plt.show()