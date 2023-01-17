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
from sklearn.linear_model import LinearRegression

import pandas as pd

import matplotlib.pyplot as plt

df = pd.read_csv("../input/insurance/insurance.csv")
df
df.head()
y= df.charges
x = df.iloc[:,0:1].values
y_predict = model.predict(x)
model = LinearRegression()

model.fit(x,y)
y_predict = model.predict(x)

y_predict
m = model.coef_

m
c = model.intercept_

c
ypredict = m*x + c

ypredict
x1 = 20

x2 = 30

w = model.predict([[x1],[x2]])

w
plt.xlabel("Age")

plt.ylabel("Charges")

plt.scatter(x,y, color = "green")

plt.plot(x,y_predict, c= "red")

plt.scatter([x1,x2], w , color= "blue")