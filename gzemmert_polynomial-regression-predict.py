# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv("/kaggle/input/polynomialregressioncsv/polynomial-regression.csv")
df.head()
df.info()
#visualize 

x=df.araba_fiyat.values.reshape(-1,1)
y=df.araba_max_hiz.values.reshape(-1,1)

plt.scatter(x,y,color="red")
plt.xlabel("araba_fiyat")
plt.ylabel("araba_max_hiz")
plt.show()


# Polynomial Regression Predict

from sklearn.preprocessing import PolynomialFeatures  

polynomial_features= PolynomialFeatures(degree=2)
x_polynomial=polynomial_features.fit_transform(x)

from sklearn.linear_model import LinearRegression

lr=LinearRegression() 

lr.fit(x_polynomial,y)

y_head2=lr.predict(x_polynomial)

plt.plot(x,y_head2,color="green")
plt.legend()
plt.show()

from sklearn.preprocessing import PolynomialFeatures  

polynomial_features= PolynomialFeatures(degree=4)
x_polynomial=polynomial_features.fit_transform(x)

from sklearn.linear_model import LinearRegression

lr=LinearRegression() 

lr.fit(x_polynomial,y)

y_head2=lr.predict(x_polynomial)

plt.plot(x,y_head2,color="green")
plt.legend()
plt.show()
#R square score of polynomial prediction model

from sklearn.metrics import r2_score
print("R_square_score:",r2_score(y,y_head2))