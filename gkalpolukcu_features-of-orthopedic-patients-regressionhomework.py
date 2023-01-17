# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns  # visualization tool

plt.style.use("seaborn-whitegrid")



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv("/kaggle/input/biomechanical-features-of-orthopedic-patients/column_2C_weka.csv")

data.head()
data.tail()
data.describe()
data.info()
data.columns
data.pelvic_incidence.value_counts()
data["class"].value_counts()
plt.scatter(data.pelvic_radius,data.sacral_slope)

plt.xlabel("Pelvic Yaricap")

plt.ylabel("Pelvic Egim")

plt.show()



from sklearn.linear_model import LinearRegression

linear_reg = LinearRegression()



x=data.pelvic_radius.values.reshape(-1,1)

y=data.sacral_slope.values.reshape(-1,1)

linear_reg.fit(x,y)



b0=linear_reg.predict([[0]])

print("b0: ",b0)

b0_ = linear_reg.intercept_

print("b0_: ",b0_) 

b1 = linear_reg.coef_

print("b1: ",b1)  



#y_head = linear_reg.predict(array).reshape(-1,1)

#plt.plot(array, y_head,color = "red")
from sklearn.linear_model import LinearRegression



x = data.iloc[:,[0,2]].values

y = data.pelvic_radius.values.reshape(-1,1)



multiple_linear_regression = LinearRegression()

multiple_linear_regression.fit(x,y)



print("b0 : ", multiple_linear_regression.intercept_)

print("b1,b2 : ", multiple_linear_regression.coef_)



multiple_linear_regression.predict(np.array([[10,40],[5,40]]))
y = data.degree_spondylolisthesis.values.reshape(-1,1)

x = data.pelvic_incidence.values.reshape(-1,1)



plt.scatter(x,y)

plt.ylabel("degree_spondylolisthesis")

plt.xlabel("pelvic_incidence")

plt.show()



# linear regression =  y = b0 + b1*x

# multiple linear regression   y = b0 + b1*x1 + b2*x2



#linear regression



from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(x,y)



y_head = lr.predict(x)

plt.plot(x,y_head,color="red")

plt.show()



print("10 Milyon tıbbi kaydın tahlili: ", lr.predict([[10000]]))



# polynomial regression =  y = b0 + b1*x +b2*x^2 + b3*x^3 + ... + bn*x^n



from sklearn.preprocessing import PolynomialFeatures

polynomial_regression = PolynomialFeatures(degree = 2)



x_polynomial = polynomial_regression.fit_transform(x)



#fit

linear_regression2 = LinearRegression()

linear_regression2.fit(x_polynomial,y)



y_head2 = linear_regression2.predict(x_polynomial)



plt.plot(x,y_head2,color= "green",label = "poly")

plt.legend()

plt.show()
x = data.iloc[:,0].values.reshape(-1,1)

y = data.iloc[:,1].values.reshape(-1,1)



#decision tree regression

from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor()   # random sate = 0

tree_reg.fit(x,y)



tree_reg.predict([[5.5]])

x_ = np.arange(min(x),max(x),0.01).reshape(-1,1)

y_head = tree_reg.predict(x_)



#visualize

plt.scatter(x,y,color="red")

plt.plot(x_,y_head,color = "green")

plt.xlabel("Pelvic Incidence")

plt.ylabel("Pelvic Egim")

plt.show()
data1 = pd.read_csv("/kaggle/input/biomechanical-features-of-orthopedic-patients/column_2C_weka.csv")



x = data1.iloc[:,0].values.reshape(-1,1)

y = data1.iloc[:,1].values.reshape(-1,1)



from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators = 10, random_state = 100)

rf.fit(x,y)



print("Yaricap değerinin Insidance orani: ",rf.predict([[5.6]]))



x_ = np.arange(min(x),max(x),0.09).reshape(-1,1)

y_head = rf.predict(x_)



# visualize

plt.scatter(x,y,color="red")

plt.plot(x_,y_head,color="green")

plt.xlabel("Pelvic Incidence")

plt.ylabel("Pelvic Orani")

plt.show()
plt.scatter(data.pelvic_radius,data.sacral_slope)

plt.xlabel("Pelvic Yaricap")

plt.ylabel("Pelvic Egim")

plt.show()

#linear regression



# sklearn library

from sklearn.linear_model import LinearRegression

# linear regression model

linear_reg = LinearRegression()



y = data.degree_spondylolisthesis.values.reshape(-1,1)

x = data.pelvic_incidence.values.reshape(-1,1)



linear_reg.fit(x,y)

y_head = linear_reg.predict(x)  # maas

plt.plot(x, y_head,color = "red")



from sklearn.metrics import r2_score

print("r_square score: ", r2_score(y,y_head))