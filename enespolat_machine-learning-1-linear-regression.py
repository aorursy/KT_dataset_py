# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import matplotlib.pyplot as plt
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data=pd.read_csv('../input/linear-regression-dataset.csv')
data.columns
data.info()
data.head(13)
plt.scatter(data.deneyim,data.maas)
plt.xlabel('deneyim')
plt.ylabel('maas')
plt.title('Deneyim Maas Cizelgesi')
plt.show()
from sklearn.linear_model import LinearRegression
linear_reg=LinearRegression()
x=data.deneyim.values.reshape(-1,1)
y=data.maas.values.reshape(-1,1)
print(type(x))
print(x.shape)
print(y.shape)

#numpy a dönüşüm yapacağız . values onu yapar

linear_reg.fit(x,y)
# prediction
#b0=linear_reg.predict(0)
b0 = linear_reg.intercept_
print("b0:",b0)


b1=linear_reg.coef_ #eğim
print("b1:",b1)
deneyim=11
maas=1663.89519747 + 1138.34819698*deneyim
print("11 yıllık deneyime göre maaşınız :",maas)
#print("11 yıllık deneyime göre maaşınız :",linear_reg.predict(11))

array=np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]).reshape(-1,1)
plt.scatter(x,y)
y_head=linear_reg.predict(array)
plt.plot(array,y_head,color='red')
#R-Square
from sklearn.metrics import r2_score
x=data.deneyim.values.reshape(-1,1)
y=data.maas.values.reshape(-1,1)
linear_reg.fit(x,y)
y_head=linear_reg.predict(x)
plt.scatter(x,y,color='green')
plt.plot(x,y_head,color='red')
print('r_square score :',r2_score(y,y_head))