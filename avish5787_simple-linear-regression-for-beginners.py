# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/tvmarketing-dataset/tvmarketing.csv')
# Display the first 5 rows
df.head()
# Display the last 5 rows
df.tail()
# Let's check the columns
df.info()
# Check the shape of the DataFrame (rows, columns)
df.shape
# Let's look at some statistical information about the dataframe.
df.describe()
# Visualise the relationship between the features and the response using scatterplots
sns.pairplot(df, x_vars=['TV'], y_vars='Sales',size=7, aspect=0.7, kind='scatter')
y = df['Sales']   #dependent variable
X = df.drop('Sales',axis = 1)   #independent variable
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=100)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
# import LinearRegression from sklearn
from sklearn.linear_model import LinearRegression

# Representing LinearRegression as lr(Creating LinearRegression Object)
lr = LinearRegression()

# Fit the model using lr.fit()
lr.fit(X_train,y_train)
# Print the intercept and coefficients
print(lr.intercept_)
print(lr.coef_)
# Making predictions on the testing set
y_pred = lr.predict(X_test)
type(y_pred)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error

lr = LinearRegression()
lr.fit(X_train,y_train)


y_test_pred = lr.predict(X_test)
y_train_pred = lr.predict(X_train)
print('r-square for train: ', r2_score(y_train,y_train_pred))
print('RMSE for train: ',np.sqrt(mean_squared_error(y_train,y_train_pred)))

print('\n')
print('r-square for test: ', r2_score(y_test,y_test_pred))
print('RMSE for test: ', np.sqrt(mean_squared_error(y_test,y_test_pred)))
plt.scatter(y_test,y_pred)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
