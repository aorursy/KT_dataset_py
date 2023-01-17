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
data1 = pd.read_csv("/kaggle/input/small-dataset-about-used-fiat-500-sold-in-italy/Used_fiat_500_in_Italy_dataset.csv")
data1.head(15)
data1.info()
data1.isnull().sum()
data1.corr()
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
data= data1.copy()
le = LabelEncoder()
data['model'] = le.fit_transform(data['model'])
data['transmission']=LabelEncoder().fit_transform(data['transmission'])

x=data1[(data1['model']=='lounge') & (data1['engine_power']==69)]['age_in_days']
y=data1[(data1['model']=='lounge') & (data1['engine_power']==69)]['price']
c=x.copy()

plt.scatter(x, y)
plt.xlabel('arac yasi')
plt.ylabel('arac fiyati')
plt.show()

from sklearn.linear_model import LinearRegression

linear_reg = LinearRegression()
x = x.values.reshape(-1,1)
y = y.values.reshape(-1,1)

linear_reg.fit(x,y)

b0 = linear_reg.predict([[5000]])
print("b0:",b0)

b0_ = linear_reg.intercept_
print("b0_:",b0_)

b1 = linear_reg.coef_
print("b1: ", b1)

print('tahmin',linear_reg.predict([[1000]]) )

array = c.values.reshape(-1,1)

plt.scatter(x,y)
y_head = linear_reg.predict(array)

plt.plot(array,y_head, color="red")

plt.show

import statsmodels.regression.linear_model as sm
lin = sm.OLS(x,y)
model = lin.fit()
model.summary()
from sklearn.model_selection import train_test_split
x=data1[(data1['model']=='lounge') & (data1['engine_power']==69)]['age_in_days']
y=data1[(data1['model']=='lounge') & (data1['engine_power']==69)]['price']

x=x.values.reshape(-1,1)
y=y.values.reshape(-1,1)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.33,random_state=0)

lr = LinearRegression()

lr.fit(x_train,y_train)

tahmin = lr.predict(x_test)

plt.scatter(x_train,y_train)
plt.plot(x_test,tahmin, color='red')
plt.show()


import statsmodels.regression.linear_model as sm
lin = sm.OLS(x_train,y_train)
model = lin.fit()
model.summary()
arac_yasi = data1[['age_in_days']]
arac_km = data1[['km']]
arac_fiyat = data1[['price']]


array = c.values.reshape(-1,1)

plt.scatter(x,y)
y_head = linear_reg.predict(array)

plt.plot(array,y_head, color="red")

plt.show
from sklearn.preprocessing import PolynomialFeatures
polynomial_regression = PolynomialFeatures(degree = 6)

x_polynomial = polynomial_regression.fit_transform(x)

linear_regression2 = LinearRegression()
linear_regression2.fit(x_polynomial,y)

y_head2 = linear_regression2.predict(x_polynomial)

plt.plot(x,y_head2,color= "green",label = "poly")
plt.legend()
plt.show()
lin = sm.OLS(x,y_head2)
model = lin.fit()
model.summary()