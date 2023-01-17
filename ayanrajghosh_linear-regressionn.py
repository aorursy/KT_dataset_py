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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import statsmodels.api as sm
Iris = pd.read_csv("../input/iris/Iris.csv")
Iris.head()
plt.figure(figsize=(16, 8))
X = Iris['SepalLengthCm'].values.reshape(-1,1)
y = Iris['PetalLengthCm'].values.reshape(-1,1)
reg = LinearRegression()
reg.fit(X, y)
print("The linear model is: Y = {:.5} + {:.5}X".format(reg.intercept_[0], reg.coef_[0][0]))
predictions = reg.predict(X)
plt.figure(figsize=(16, 8))
plt.scatter(
    Iris['SepalLengthCm'],
    Iris['PetalLengthCm'],
    c='black'
)
plt.plot(
    Iris['SepalLengthCm'],
    predictions,
    c='blue',
    linewidth=2
)
plt.xlabel("SepalLength")
plt.ylabel("PetalLength")
plt.show()
X = Iris['SepalLengthCm']
y = Iris['PetalLengthCm']
X2 = sm.add_constant(X)
est = sm.OLS(y, X2)
est2 = est.fit()
print(est2.summary())
