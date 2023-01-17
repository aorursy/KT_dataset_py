# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/50_Startups.csv')

data.head()
X = data.iloc[:, 0:4]

X.head()
y = data.iloc[:,4]

y.head()
X = pd.get_dummies(X)

X.head()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_test.head()
from sklearn.linear_model import LinearRegression

regg = LinearRegression()
regg.fit(X_train, y_train)
y_pred = regg.predict(X_test)
y_res = pd.Series(y_pred)
y_res.head()
from sklearn.metrics import r2_score

r2_score(y_test,y_res)
data.head()
plt.plot(data.iloc[:,0], data.iloc[:, 4])

plt.xlabel('R&D Spends')

plt.ylabel('Profits')