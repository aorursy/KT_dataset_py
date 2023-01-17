# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data1=pd.read_csv("../input/biomechanical-features-of-orthopedic-patients/column_2C_weka.csv")
data1.head()
data1.describe()
x=data1.pelvic_incidence.values.reshape(-1,1)

y=data1.sacral_slope.values.reshape(-1,1)
plt.figure(figsize=(10,10))

plt.scatter(x=x,y=y,color='black')

plt.xlabel("PelvicIncidence")

plt.ylabel("SacralSlope")

plt.show()
from sklearn.linear_model import LinearRegression
lr=LinearRegression()



#fitting process

lr.fit(x,y)
y_head=lr.predict(x)


plt.figure(figsize=(10,10))

plt.scatter(x=x,y=y,color='black')

plt.xlabel("PelvicIncidence")

plt.ylabel("SacralSlope")

plt.plot(x,y_head)

plt.show()

print('R^2 score: ',lr.score(x, y))