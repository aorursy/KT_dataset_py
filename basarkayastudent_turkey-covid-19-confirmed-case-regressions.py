# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data=pd.read_csv("/kaggle/input/uncover/UNCOVER_v4/UNCOVER/johns_hopkins_csse/johns-hopkins-covid-19-daily-dashboard-cases-over-time.csv")
data
dft=data[data.country_region=="Turkey"]

dft
list(dft.index.values)
days=[]

for i in list(dft.index.values):

    days.append(i-20760)

days
dft["days"]=days
x=dft.days.values.reshape(-1,1)

y=dft.confirmed.values.reshape(-1,1)

f,ax=plt.subplots(figsize=(20,9))

plt.scatter(x, y, label="Data")

plt.xlabel("Days")

plt.ylabel("Confirmed Cases")

plt.legend()

plt.show()
# Linear Regression

from sklearn.linear_model import LinearRegression

lin_reg=LinearRegression()

lin_reg.fit(x,y)



y_head_lin=lin_reg.predict(x)

f,ax=plt.subplots(figsize=(20,9))

plt.scatter(x,y, label="Data")

plt.plot(x, y_head_lin, label="Linear Regression", color="red")

plt.legend()

plt.show()
# Polynomial Regression

from sklearn.preprocessing import PolynomialFeatures

poly_tra=PolynomialFeatures(degree=6)

x_poly=poly_tra.fit_transform(x)



poly_reg=LinearRegression()

poly_reg.fit(x_poly, y)



y_head_poly=poly_reg.predict(x_poly)



f,ax=plt.subplots(figsize=(20,9))

plt.scatter(x,y,label="Data")

plt.plot(x, y_head_lin, label="Linear Regression", color="red")

plt.plot(x, y_head_poly, label="Polynomial Regression", color="orange")

plt.legend()

plt.xlabel("Days")

plt.ylabel("Confirmed Cases")

plt.show()
# Decision Trees Regression

from sklearn.tree import DecisionTreeRegressor



dt_reg=DecisionTreeRegressor()

dt_reg.fit(x,y)



y_head_dt=dt_reg.predict(x)



f,ax=plt.subplots(figsize=(20,9))

plt.scatter(x,y,label="Data")

plt.plot(x, y_head_lin, label="Linear Regression", color="red")

plt.plot(x, y_head_poly, label="Polynomial Regression", color="orange")

plt.plot(x, y_head_dt, label="Decision Trees Regression", color="green")

plt.legend()

plt.xlabel("Days")

plt.ylabel("Confirmed Cases")

plt.show()
# Random Forest Regression

from sklearn.ensemble import RandomForestRegressor



rf_reg=RandomForestRegressor(n_estimators=100, random_state=42)

rf_reg.fit(x,y)



y_head_rf=rf_reg.predict(x)



f,ax=plt.subplots(figsize=(20,9))

plt.scatter(x,y,label="Data")

plt.plot(x, y_head_lin, label="Linear Regression", color="red")

plt.plot(x, y_head_poly, label="Polynomial Regression", color="orange")

plt.plot(x, y_head_dt, label="Decision Trees Regression", color="green")

plt.plot(x, y_head_rf, label="Random Forest Regression", color="purple")

plt.legend()

plt.xlabel("Days")

plt.ylabel("Confirmed Cases")

plt.show()
input1=float(input("Days Since 20th of May: "))+120

array1=np.array([[float(input1)]])

poly_input=poly_tra.fit_transform(array1)



print("Linear Regression:", int(round(lin_reg.predict([[float(input1)]])[0][0])), "Confirmed Cases")

print("Polynomial Regression:", int(round(poly_reg.predict(poly_input)[0][0])), "Confirmed Cases")

print("Decision Trees Regression:", int(round(dt_reg.predict([[float(input1)]])[0])), "Confirmed Cases")

print("Random Forest Regression:", int(round(rf_reg.predict([[float(input1)]])[0])), "Confirmed Cases")