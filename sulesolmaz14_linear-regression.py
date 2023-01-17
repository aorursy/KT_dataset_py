# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#import data
df=pd.read_csv("../input/yenicsv/yeniveriseti.csv", sep=";")
df.head()

#plot data
plt.scatter(df.deneyim,df.maas)
plt.xlabel("deneyim")
plt.ylabel("maas")
plt.show()

 #Linear Regression model
linear_reg=LinearRegression()
x=df.deneyim.values.reshape(-1,1)
y=df.maas.values.reshape(-1,1)
linear_reg.fit

type(x)
x.shape
linear_reg.fit(x,y)
#prediction
b0=linear_reg.predict([[0]])
print("b0: ", b0)
b0_=linear_reg.intercept_
print("b0: ", b0_)#y eksenini kestiği nokta yai intersept
b1 =linear_reg.coef_
print("b1: ", b1)#eğim(slope)
#maas=163+1138*deneyim
maas_yeni=1663+1138*11
print(maas_yeni)
#prediction
print(linear_reg.predict([[11]]))
#visualize line
array=np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]).reshape(-1,1)
plt.scatter(x,y)
y_head=linear_reg.predict(array)
plt.plot(array,y_head,color="red")
plt.show()

linear_reg.predict([[100]])
#Multiple Linear regression= b0 +b1*x1 +b2*x2...
#
df=pd.read_csv("../input/multiple-linear-regression/multiple_regression_dataset.csv", sep=";")
df.head()
x=df.iloc[:,[0,2]].values
y=df.maas.values.reshape(-1,1)

multiple_linear_regression = LinearRegression()
multiple_linear_regression.fit(x,y)

print("b0: ", multiple_linear_regression.intercept_)
print("b1,b2: ", multiple_linear_regression.coef_)
multiple_linear_regression.predict(np.array([[10,35], [5,35]]))
df= pd.read_csv("../input/polynomial-linear-regression/data.csv", sep =";")

y= df.araba_max_hiz.values.reshape(-1,1)
x= df.araba_fiyat.values.reshape(-1,1)

plt.scatter(x,y)
plt.ylabel("araba_max_hiz")
plt.xlabel("araba_fiyat")
plt.show()
#linear regression = y=b0+b1*x
#multiple linear regression y = b0+b1*x1+b2*x2

#%%Linear Regression
from sklearn.linear_model import LinearRegression

lr =LinearRegression()
lr.fit(x,y)
#%% predict

y_head = lr.predict(x)
plt.scatter(x, y)
plt.plot(x, y_head, color ="red", label="linear")
plt.show()

print("10 milton tl' lik araba hızı tahmini :" ,lr.predict([[10000]]))
#polynomial linear regression = y= b0+ b1*x +b2*x^2+b3*x^3+......
from sklearn.preprocessing import PolynomialFeatures
polynomial_regression =PolynomialFeatures(degree= 4)

x_polynomial = polynomial_regression.fit_transform(x)
#fit
linear_regression2 =LinearRegression()
linear_regression2.fit(x_polynomial, y)

#görselleştirme
y_head2 = linear_regression2.predict(x_polynomial)
plt.scatter(x, y)
plt.plot(x,y_head2, color="green", label="poly")
plt.legend()
plt.show()