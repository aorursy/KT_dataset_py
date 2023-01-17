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
df1=pd.read_csv('../input/column_2C_weka.csv')

df2=pd.read_csv('../input/column_3C_weka.csv')
df1.head()
#EKSİK VERİMİZ VARMI DİYE  KONTROL ETTİK 
df1.pelvic_incidence.value_counts()

df1.sacral_slope.values

df1.info()
x=df1.iloc[:,0].values.reshape(-1,1)

#print(x)

y=df1.iloc[:,3].values.reshape(-1,1)

#print(y)
#SCATTER PLOT
plt.figure(figsize=(10,10))

plt.scatter(x,y,color='red')

plt.xlabel('pelvic_incidence')

plt.ylabel('pelvic_radius')

plt.show()
#linear regression modelimizi uygulayalım
from sklearn.linear_model import LinearRegression

linear_reg1=LinearRegression()

linear_reg1.fit(x,y)



b0=linear_reg1.predict([[0]])

b1=linear_reg1.coef_



x_=np.arange(min(x),max(x),0.01).reshape(-1,1)

y_head=linear_reg1.predict(x_).reshape(-1,1)



plt.figure(figsize=(10,10))

plt.scatter(x,y,color='red')

plt.plot(x_,y_head,color='green')

plt.xlabel('pelvic_incidence')

plt.ylabel('sacral_slope')

plt.show()



print(linear_reg1.predict([[28.5]]))
from sklearn.metrics import r2_score



print('r2_score:',r2_score(x_,y_head))
df2.head()
x2=df2.sacral_slope.values.reshape(-1,1)

y2=df2.pelvic_radius.values.reshape(-1,1)

plt.figure(figsize=(10,10))

plt.scatter(x2,y2,color='red')

plt.xlabel('sacral_slope')

plt.ylabel('pelvic_radius')

plt.show()
lr=LinearRegression()

lr.fit(x2,y2)



b0=lr.predict([[0]])

b1=lr.coef_



x_2=np.arange(min(x),max(x),0.01).reshape(-1,1)

y_2head=lr.predict(x_2)



plt.figure(figsize=(10,10))

plt.scatter(x2,y2,color='red')

plt.plot(x_2,y_2head,color='green')

plt.xlabel('sacral_slope')

plt.ylabel('pelvic_radius')

plt.show()

lr.predict([[28.5]])
print(r2_score(x_2,y_2head))
x3=df1.iloc[:,0].values.reshape(-1,1)

y3=df1.iloc[:,3].values.reshape(-1,1)





from sklearn.preprocessing import PolynomialFeatures

p_lr=PolynomialFeatures(degree=4)



x_poly=p_lr.fit_transform(x3)

linear_regression4=LinearRegression()

linear_regression4.fit(x_poly,y3)



y_head4=linear_regression4.predict(x_poly)

plt.scatter(x3,y3,color='red')

plt.plot(x3,y_head4,color='green')

plt.show()

linear_regression4.predict([[28]])