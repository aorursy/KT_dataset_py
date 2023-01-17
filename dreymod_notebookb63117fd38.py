# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, Lasso

import statsmodels.api as sm

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv("../input/brent-oil-prices/BrentOilPrices.csv")

data.head()
index=1000#len(data.to_numpy())

x=np.linspace(1, index, index)

y=np.zeros(index)

x=x.reshape((-1,1))

data1=data.to_numpy()

for i in range(index):

    y[i]=data1[i][1]
model = LinearRegression().fit(x, y)

y_pr = model.predict(x)

print('coefficient of determination:', model.score(x, y))

print('intercept:', model.intercept_)

print('slope:', model.coef_,'\n')

print('\n','\n','\n')
plt.plot(x,y,'b.')

plt.plot(x,y_pr,'r')

plt.xlabel('Month')

plt.ylabel('Price, $')

plt.title('Brent Oil Prices')

plt.grid(True)

plt.show()
model_lasso=Lasso(alpha=0.5).fit(x,y)

y_pr = model_lasso.predict(x)

print('coefficient of determination:', model_lasso.score(x, y))

print('intercept:', model_lasso.intercept_)

print('slope:', model_lasso.coef_,'\n')
plt.plot(x,y,'b.')

plt.plot(x,y_pr,'r')

plt.xlabel('Month')

plt.ylabel('Price, $')

plt.title('Brent Oil Prices')

plt.grid(True)

plt.show()