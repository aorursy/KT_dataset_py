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

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')
#Importing the dataset

dataset=pd.read_csv('../input/position-salaries/Position_salaries.csv')
dataset.head(n=10)
dataset.info() #data type
dataset.describe()
dataset.isnull().sum()#null values
#Seperating the X and Y columns

X=dataset.iloc[:,1:2].values

Y=dataset.iloc[:,2].values
# Fitting Linear Regression to the dataset

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()

lin_reg.fit(X, Y)
# Fitting Polynomial Regression to the dataset

from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree = 3)

X_poly = poly_reg.fit_transform(X)

poly_reg.fit(X_poly, Y)

lin_reg_2 = LinearRegression()

lin_reg_2.fit(X_poly, Y)
# Visualising the Linear Regression results

plt.scatter(X, Y, color = 'red')

plt.plot(X, lin_reg.predict(X), color = 'blue')

plt.show()
# Visualising the Polynomial Regression results

plt.scatter(X, Y, color = 'red')

plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')

plt.title('Truth or bluff(Polynomial Regression)')

plt.xlabel('position level')

plt.ylabel('salary')

plt.show()
# Predicting a new result with Linear Regression

lin_reg.predict([[6.5]])
# Predicting a new result with Polynomial Regression

lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))