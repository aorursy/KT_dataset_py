# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import seaborn

import matplotlib.pyplot as plt

import pylab as pl

from sklearn.linear_model import LinearRegression

%matplotlib inline



seaborn.set()
# Creat a smiple data set

np.random.seed(0)

X = np.random.random(size=(20,1))

#y = 3*X.ravel() +2 + np.random.randn(20)   

#np.squeeze()? , 和 ravel 类似？

y = 3*X.squeeze() +2 + np.random.randn(20)

print(y)
plt.plot(X, y , "o")
model = LinearRegression()

model.fit(X, y)



X_fit = np.linspace(0, 1, 100)[:, np.newaxis]

y_fit = model.predict(X_fit)

y_model = 3*X_fit.ravel() + 2 +np.random.randn(100)



plt.plot(X, y ,'o')

plt.plot(X_fit, y_fit)

plt.plot(X_fit, y_model)