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
dataset = pd.read_csv("../input/polynomialregression/datasets_20610_26621_Position_Salaries.csv")
dataset
dataset.head()
import matplotlib.pyplot as plt
plt.plot(dataset["Level"],dataset["Salary"],color="green")
plt.scatter(dataset["Level"],dataset["Salary"],color="green")
plt.show()
dataset.isnull().any()
x=dataset.iloc[:,1:2]
y=dataset.iloc[:,2:3]
x
y
x=x.values
y=y.values
x
y
#apply Linear Regression

from sklearn.linear_model import LinearRegression
lregressor = LinearRegression()
lregressor.fit(x,y)
ypredict=lregressor.predict(x)
ypredict
y
plt.scatter(x,y)
plt.plot(x,ypredict)
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=4)
x_poly=poly_reg.fit_transform(x)
x_poly
y
poly_reg.fit(x_poly,y)
polynomreg=LinearRegression()
polynomreg.fit(x_poly,y)
polypred=polynomreg.predict(x_poly)
plt.scatter(x,y)
plt.scatter(x,y)
plt.plot(x,polypred)
from sklearn.metrics import r2_score
lraccuracy = r2_score(y,ypredict)
lraccuracy
polyaccuracy = r2_score(y,polypred)
polyaccuracy
pwd

polynomreg.predict([[1,2,4,8,16]]) # 2 pow 0 2 pow 1------2 pow 4
po=poly_reg.fit_transform([[2]])
polynomreg.predict(po)
polynomreg.predict([[1,20,400,8000,160000]])  # powers of 20
po1=poly_reg.fit_transform([[20]])
polynomreg.predict(po1)
