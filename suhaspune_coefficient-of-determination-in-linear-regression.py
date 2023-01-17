# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
# Importing the dataset

dataset = pd.read_csv('/kaggle/input/years-of-experience-and-salary-dataset/Salary_Data.csv')

X = dataset.iloc[:, 0].values

y = dataset.iloc[:, 1].values
# As we need 2D Input 

X=X.reshape(-1, 1)
X.shape
y.shape
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(x_train, y_train)
m=lr.coef_

b=lr.intercept_

print("slope=",m, "intercept=",b)
y_pred = lr.predict(x_test)
plt.scatter(x_train, y_train, color = "blue")

plt.plot(x_train, lr.predict(x_train), color = "red")

plt.title("Salary vs Experience")

plt.xlabel("Years of Experience")

plt.ylabel("Salary")

plt.show()
from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
# variance score: 1 is perfect prediction

from sklearn.metrics import mean_squared_error, r2_score

print('Variance score: %.2f' % r2_score(y_test, y_pred))
print(lr.predict([[6]]))