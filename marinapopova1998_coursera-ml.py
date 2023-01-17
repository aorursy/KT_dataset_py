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
data = pd.read_csv("/kaggle/input/coursera-ml/ex1data1.txt",

                  header=None,names=["Population","Profit"])

data.head()
import matplotlib.pyplot as plt

%matplotlib inline
x = data["Population"]

y = data["Profit"]
plt.plot(x,y,'rx');

plt.grid();

Y = 1.2*x -3

plt.plot(x,Y,'b-');
from sklearn.linear_model import LinearRegression
lr = LinearRegression()

x_r = np.array(x).reshape(-1, 1)

lr.fit(x_r,y)
lr.score(x_r,y)
plt.plot(x,y,'rx');

plt.grid();

Y = x*lr.coef_[0] + lr.intercept_

plt.plot(x,Y,'b-');
lr.predict(np.array([20]).reshape(1,-1))
data2 = pd.read_csv("/kaggle/input/coursera-ml/ex1data2.txt",

                   header=None, names=["square","rooms","price"])

data2.head()
from sklearn.preprocessing import StandardScaler
x = data2.drop("price", axis=1)

y = data2["price"]
scaler = StandardScaler()

x_scaled = scaler.fit_transform(x)
lr = LinearRegression()

lr.fit(x_scaled,y)
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LinearRegression
scaler = StandardScaler()

x = data2.drop("price", axis=1)

y = data2["price"]
scaler.fit(x)

x_scaled = scaler.fit_transform(x)
lr = LinearRegression()

lr.fit(x_scaled, y)

print(lr.score(x_scaled,y))