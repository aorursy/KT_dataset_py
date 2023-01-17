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
data = pd.read_csv("/kaggle/input/world-happiness/2018.csv")
data.head()

data.info()
y = data.Score.values.reshape(-1,1)

x = data["GDP per capita"].values.reshape(-1,1)
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(x,y)

y_head = lr.predict(x)

plt.plot(x,y_head,color='red')



plt.scatter(x,y)

plt.ylabel("Happines Score")

plt.xlabel("GDP Per Capita")

plt.show()
from sklearn.metrics import r2_score

r2_score(y,y_head)
from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor(n_estimators=100,random_state=42)

rfr.fit(x,y)

y_head_ = rfr.predict(x)

plt.scatter(x,y)

plt.plot(x,y_head_,color='g')

plt.ylabel("Happines Score")

plt.xlabel("GDP Per Capita")

plt.show()
r2_score(y,y_head_)
def whichmodelisbetter(x,y):

    lr = LinearRegression()

    lr.fit(x,y)

    y_head = lr.predict(x)

    rfr.fit(x,y)

    y_head_ = rfr.predict(x)

    lr2 = r2_score(y,y_head)

    rr2 = r2_score(y,y_head_)

    if lr2 > rr2:

        print("Linear Regression Model is better")

    else:

        print("Random Forest Regression Model is better")
whichmodelisbetter(x,y)