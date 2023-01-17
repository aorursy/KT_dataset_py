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
data = pd.read_csv("/kaggle/input/coursera-ml/ex1data1.txt", header=None, names=["Population", "Profit"])
data.head()
import matplotlib.pyplot as plt

%matplotlib inline
x = data["Population"]

y = data["Profit"]
k = 0.4

b = 0

h = k*x + b
plt.figure(figsize=(10,5))

plt.plot(x,y, "rx");

plt.plot(x,h);

plt.grid();
from sklearn.linear_model import LinearRegression
lr = LinearRegression()

lr.fit(np.array(x).reshape(-1,1),y)
lr.score(np.array(x).reshape(-1,1),y)
print(lr.coef_)

print(lr.intercept_)

print(lr.rank_)
h = lr.coef_*x + lr.intercept_
plt.figure(figsize=(10,5))

plt.plot(x,y, "rx");

plt.plot(x,h);

plt.grid();
lr.predict(np.array([20]).reshape(1, -1))
data2 = pd.read_csv("/kaggle/input/coursera-ml/ex1data2.txt", header=None, names=["square", "rooms", "price"])
data2.head()
x = data2.drop("price",axis=1)

y = data2["price"]

lr = LinearRegression()

lr.fit(x,y);
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x = data2.drop("price", axis = 1)

y = data2["price"]
x_scaled = scaler.fit_transform(x)
lr2 = LinearRegression()

lr2.fit(x_scaled, y)
lr2.score(x_scaled, y)
lr2.predict(scaler.transform([[2002,3],[3004, 4]]))