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

df=pd.read_csv("../input/housesalesprediction/kc_house_data.csv")

df.head()
df.columns
df.dropna(axis=0)
y=df['price']

x=df[['bedrooms', 'bathrooms', 'sqft_living','sqft_lot']]

x.head()
y.head()
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33)

print(x_train.shape)

print(y_train.shape)

print(x_test.shape)

print(y_test.shape)
from sklearn.linear_model import LinearRegression





Lreg=LinearRegression()

Lreg.fit(x_train,y_train)
y_pred=Lreg.predict(x_test)

print(y_pred)

from sklearn.metrics import r2_score

print(r2_score(y_test,y_pred))
import matplotlib.pyplot as plt

%matplotlib inline

from mpl_toolkits.mplot3d import Axes3D



fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')



x = x_test['sqft_living']

y =  x_test['sqft_lot']

z =  x_test['bedrooms']

c = x_test['bathrooms']



img = ax.scatter(x, y, z, c=c, cmap=plt.hot())

fig.colorbar(img)

plt.show()
import seaborn as sns



ax1 = sns.distplot(y_test, hist=False, color="r", label="Actual Value")

sns.distplot(y_pred, hist=False, color="b", label="Fitted Values" , ax=ax1)