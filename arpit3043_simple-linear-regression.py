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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import statsmodels.api as sm

import seaborn as sns

sns.set()
dataset = pd.read_csv('/kaggle/input/random-salary-data-of-employes-age-wise/Salary_Data.csv')

x = dataset.iloc[:, :-1].values

y = dataset.iloc[:, :-1].values
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 1/3, random_state = 0)
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_test)
##declaring independent and dependent variables

y = dataset['Salary']

x1 = dataset['YearsExperience']
##producing a scatter plot to explore the data

plt.scatter(x1,y)

plt.xlabel('YearsExperience',fontsize=20)

plt.ylabel('Salary',fontsize=20)

plt.show()
##creating a linear regression

x = sm.add_constant(x1)

results = sm.OLS(y,x).fit()

results.summary()
##now plotting the regression line on the scatter plot

plt.scatter(x1,y)

yhat = 9449.96*x1+25790

fig = plt.plot(x1,yhat,lw=2,c='orange')

plt.xlabel('Years of Experience',fontsize=20)

plt.ylabel('Salary',fontsize=20)

plt.show()
plt.scatter(x_test, y_test, color = 'green')

plt.plot(x_train, regressor.predict(x_train), color = 'orange')

plt.title('salary vs experience(training set)')

plt.xlabel('years of experience')

plt.ylabel('salary')

plt.show()