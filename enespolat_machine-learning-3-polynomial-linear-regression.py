# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data=pd.read_csv('../input/polynomial-regression.csv')
data.head(9)
y=data.araba_max_hiz.values.reshape(-1,1)
x=data.araba_fiyat.values.reshape(-1,1)#sklearn için gerekli
plt.scatter(x,y)
plt.ylabel('araba_max_hiz')
plt.xlabel('araba_fiyat')
plt.show()
#örneğin burda linear regression deneyelim
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x,y)
y_head=lr.predict(x)
plt.scatter(x,y)
plt.plot(x,y_head,color='red')
plt.show()

#bu predict doğru olmadı
#Polynomial predict
from sklearn.preprocessing import PolynomialFeatures
polynomial_regression=PolynomialFeatures(degree=4)
x_polynomial=polynomial_regression.fit_transform(x)

linear_regression2=LinearRegression()
linear_regression2.fit(x_polynomial,y)
y_head2=linear_regression2.predict(x_polynomial)
plt.scatter(x,y)
plt.plot(x,y_head2,color='green',label='poly')
plt.plot(x,y_head,color='red',label='linear')
plt.legend()
plt.show()

