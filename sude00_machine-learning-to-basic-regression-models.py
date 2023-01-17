# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

plt.style.use("seaborn-whitegrid")



import seaborn as sns



from collections import Counter



import warnings

warnings.filterwarnings("ignore")



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import matplotlib.pyplot as plt

df= pd.read_csv('/kaggle/input/biomechanical-features-of-orthopedic-patients/column_2C_weka.csv')

plt.figure(figsize=[10,10])

plt.scatter(df.pelvic_incidence,df.sacral_slope)

plt.xlabel("pelvic incidence")

plt.ylabel("sacral slope")

from sklearn.linear_model import LinearRegression

linear_reg=LinearRegression()

x=df.pelvic_incidence.values.reshape(-1,1)

y=df.sacral_slope.values.reshape(-1,1)

linear_reg.fit(x,y)

y_head=linear_reg.predict(x)
#%% # Plot regression line and scatter

plt.figure(figsize=[10,10])

plt.scatter(x,y)

plt.xlabel("pelvic incidence")

plt.ylabel("sacral slope")

plt.plot(x,y_head,color="red")

plt.show()
#%% prediction

b0=linear_reg.predict([[0]])

b1=linear_reg.coef_

print("b0: ",b0)

print("b1: ",b1)

#pelvic incidence = 80 sacral slope=?

print("Predict: ",b1*80+b0)
#%% R^2 score

from sklearn.metrics import r2_score

print("R^2 : ",r2_score(y,y_head))



from sklearn.linear_model import LinearRegression



x=df.iloc[:,[1,2,3,4,5]].values

y=df.pelvic_incidence.values.reshape(-1,1)



multiple_reg=LinearRegression()

multiple_reg.fit(x,y)

print("b0: ",multiple_reg.intercept_) # or print("b0: ",multiple_reg.predict(0))

print("b1,b2,b3,b4,b5 : ", multiple_reg.coef_)
y_head=multiple_reg.predict(x)

#%% R^2 score

from sklearn.metrics import r2_score

print("R^2 : ",r2_score(y,y_head))
#%% R^2 score

print("R^2 :",multiple_reg.score(x,y))
from sklearn.preprocessing import PolynomialFeatures

polynomial_regression = PolynomialFeatures(degree = 4)

x_polynomial = polynomial_regression.fit_transform(x)
from sklearn.linear_model import LinearRegression

plt.figure(figsize=[20,20])

poly_reg=LinearRegression()

poly_reg.fit(x_polynomial,y)

y_head=poly_reg.predict(x_polynomial)

plt.plot(x,y_head,color="orange")

plt.show()
#%% R^2 score

from sklearn.metrics import r2_score

print("R^2 : ",r2_score(y,y_head))
from sklearn.tree import DecisionTreeRegressor

import numpy as np

x=df.pelvic_incidence.values.reshape(-1,1)

y=df.sacral_slope.values.reshape(-1,1)

tree=DecisionTreeRegressor()

tree.fit(x,y)



x_=np.arange(min(x),max(x),0.01).reshape(-1,1)

y_head=tree.predict(x_)

plt.plot(x_,y_head,color="red")

plt.show()
x=df.pelvic_incidence.values.reshape(-1,1)

y=df.sacral_slope.values.reshape(-1,1)

plt.figure(figsize=[20,20])

plt.scatter(x,y)



from sklearn.ensemble import RandomForestRegressor



random_reg=RandomForestRegressor(n_estimators=100,random_state=42)



random_reg.fit(x,y)



x_=np.arange(min(x),max(x),0.01).reshape(-1,1)

y_head=random_reg.predict(x_)

plt.plot(x_,y_head,color="red")

plt.show()