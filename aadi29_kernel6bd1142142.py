# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
make_regression()

bias=100
X,y,coef = make_regression(n_samples=506, n_features=1,noise=5,random_state=42, bias = bias,coef=True)
X.shape , y.shape

X
y


plt.scatter(X,y)
y_new = X*coef + bias

y_new
plt.scatter(X,y)
plt.plot(X,y_new,'r')
from sklearn.linear_model import LinearRegression
model =LinearRegression()
model.fit(X,y)
model.coef_
model.intercept_
plt.scatter(X,y)
plt.plot(X,y_new,'r',label="Pre")
plt.plot(X,model.predict(X),'y',label="LR")
plt.legend()
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test  = train_test_split(X,y,test_size=0.33,random_state=42)
model.fit(X_train,y_train)
model.predict(X_test[:10])
y_test[:10]
model.score(X_test,y_test)
