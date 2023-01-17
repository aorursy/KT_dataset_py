# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # visualization



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv("/kaggle/input/insurance/insurance.csv")
df.info
df.head()
x=df.bmi

y=df.charges
# Görselleştirme

plt.scatter(x,y,color="red")

plt.xlabel("bmi")

plt.ylabel("charges")

plt.show()
# linear regression



from sklearn.linear_model import LinearRegression



reg=LinearRegression()
x=df.bmi.values.reshape(-1,1)

y=df.charges.values.reshape(-1,1)



reg.fit(x,y)
b0=reg.intercept_

b1=reg.coef_

print(b0)

print(b1)
x_=np.linspace(min(x),max(x)).reshape(-1,1)

y_head=reg.predict(x_)
# Görselleştirme

plt.scatter(x,y,color="red")

plt.plot(x_,y_head,color="black",linewidth=4)

plt.xlabel("bmi")

plt.ylabel("charges")

plt.show()
# Tahmin yapalım

print(reg.predict([[32]]))
# age ve bmi

x=df.iloc[:,[0,2]].values

y=df.charges.values.reshape(-1,1)
multiple_reg=LinearRegression()

multiple_reg.fit(x,y)
b0=multiple_reg.intercept_

b1=multiple_reg.coef_ # b1,b2 

print(b0)

print(b1)
multiple_reg.predict(np.array([[18,31.000],[50,30.000]]))
x=df.bmi.values.reshape(-1,1)

y=df.charges.values.reshape(-1,1)
plt.scatter(x,y,color="red")
from sklearn.preprocessing import PolynomialFeatures



polynomial_reg=PolynomialFeatures(degree=2)



x_polynomial=polynomial_reg.fit_transform(x) # x^2'e kadar oluşturduk.
polynomial_reg2=LinearRegression()



polynomial_reg2.fit(x_polynomial,y) 
y_head=polynomial_reg2.predict(x_polynomial)
# Görselleştirme

plt.scatter(x,y,color="red")

plt.plot(x,y_head,color="blue")

plt.xlabel("bmi")

plt.ylabel("charges")

plt.show()
x=df.iloc[:,2].values.reshape(-1,1)

y=df.iloc[:,6].values.reshape(-1,1)
from sklearn.tree import DecisionTreeRegressor,plot_tree



tree_reg=DecisionTreeRegressor(random_state=0)
tree_reg.fit(x,y)
x_=np.arange(min(x),max(x),0.01).reshape(-1,1)

y_head=tree_reg.predict(x_)

plt.scatter(x,y,color="red")

plt.plot(x_,y_head,color="green")

plt.show()
# Tahmin yapalım

tree_reg.predict([[52]])