import matplotlib.pyplot as plt

import seaborn as sns
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
import os

for dirname,_,filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname,filename))
wine = pd.read_csv('/kaggle/input/predict-red-wine-quality/train.csv')
wine.head()
sns.pairplot(wine)

plt.show()
corrs = wine.corr()

corrs
x = wine[['quality']]

y  = wine[['alcohol']]
from sklearn.model_selection import train_test_split

x_train ,x_test, y_train , y_test  = train_test_split(x,y,test_size=0.3,random_state=0)
x_train.shape , x_test.shape
from sklearn.linear_model import LinearRegression

slr = LinearRegression()

slr.fit(x_train , y_train)
slr.intercept_ , slr.coef_
y_pred = slr.predict(x_train)
y_pred
plt.scatter(x_train,y_train,color = 'red')

plt.plot(x_train,slr.predict(x_train))

plt.xlabel('alcohol')

plt.ylabel('quantity')

plt.show()
from sklearn.metrics import r2_score
r2_score(y_train,y_pred)
x1 = wine[['alcohol','citric.acid']]
y1 = wine[['quality']]
x1_train ,x1_test , y1_train ,y1_test = train_test_split(x1,y1,test_size = 0.3,random_state = 0)
x1_train.shape , x1_test.shape
from sklearn.linear_model import LinearRegression
mlr = LinearRegression()

mlr.fit(x1_train , y1_train)
mlr.intercept_ , mlr.coef_

y_mlr_pred = mlr.predict(x1_train)
r2_score(y1_train , y_mlr_pred)
x3 = wine[['alcohol','citric.acid','pH']]

y3 = wine[['quality']]
x3_train ,x3_test , y3_train ,y3_test = train_test_split(x3,y3,test_size = 0.3,random_state = 0)
mlr1 = LinearRegression()

mlr1.fit(x3_train , y3_train)
mlr.intercept_ , mlr.coef_
y_mlr_pred1 = mlr1.predict(x3_train)
r2_score(y3_train,y_mlr_pred1)