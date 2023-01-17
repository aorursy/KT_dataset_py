# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import warnings  

warnings.filterwarnings("ignore")   # ignore warnings



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Load the data from csv file

data = pd.read_csv('../input/Admission_Predict_Ver1.1.csv')
data.head()
data.tail()
data.info()
data.describe()
data.shape
data.columns
data.isnull()
# Correlation

data.corr()
data.info()
GRE = data.iloc[:, 1:2]

GRE
admission = data.iloc[:, -1:]

admission
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(GRE, admission, test_size = 0.33,random_state = 0)
x_train
y_train
x_test
y_test
data.head()
from sklearn.linear_model import LinearRegression 
# Creation of model

# The model will learn y_train from x_train.

lr = LinearRegression()

lr.fit(x_train, y_train)
# Prediction

prediction = lr.predict(x_test)

prediction
# Visualization without scaling

plt.plot(x_test, prediction, color='black', linewidth=3)

plt.scatter(x=GRE,y=admission)

plt.xlabel('GRE Score')

plt.ylabel('Chance of Admission')

plt.show()
toefl = data.iloc[:,2:3]

toefl
x_train1,x_test1,y_train1,y_test1 = train_test_split(toefl, admission, test_size = 0.33,random_state = 0)
x_train1
y_train1
x_test1
y_test1
# Creation of model

lr1 = LinearRegression()

lr1.fit(x_train1, y_train1)
# Prediction

prediction1 = lr1.predict(x_test1)

prediction1
plt.plot(x_test1, prediction1, color='black', linewidth=3)

plt.scatter(x=toefl,y=admission)

plt.xlabel('Toefl Score')

plt.ylabel('Chance of Admission')

plt.show()
data.head()
part1 = data.iloc[:, 1:4]

part1
gpa = data.iloc[:,6:7]

gpa
part2 = pd.concat([part1, gpa], axis=1)

part2
x_train2,x_test2,y_train2,y_test2 = train_test_split(part2, admission, test_size = 0.33,random_state = 0)
x_train2
x_test2
y_train2
y_test2
# Creation of model

regressor = LinearRegression()

regressor.fit(x_train2, y_train2)



# Prediction

prediction2 = regressor.predict(x_test2)

prediction2
toefl
admission
toefl.shape
admission.shape
TOEFL = toefl.values

ADMISSION = admission.values
from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree=2)  #second degree polynomial regression

toefl_poly = poly_reg.fit_transform(TOEFL)

print(toefl_poly)
# Creation of model

lin_reg2 = LinearRegression()

lin_reg2.fit(toefl_poly,admission)



# Prediction

plt.scatter(toefl, admission, color='red')

plt.plot(TOEFL, lin_reg2.predict(poly_reg.fit_transform(TOEFL)))

plt.show()
# Scaling

from sklearn.preprocessing import StandardScaler

sc1 = StandardScaler()

toefl_scaled = sc1.fit_transform(toefl)

sc2 = StandardScaler()

admission_scaled = sc2.fit_transform(admission)
from sklearn.svm import SVR
# Creation of model

svr_reg = SVR(kernel='rbf')  #radial basis function

svr_reg.fit(toefl_scaled,admission_scaled)
# Prediction

prediction4 = svr_reg.predict(toefl_scaled)

prediction4
# import the library

from sklearn.tree import DecisionTreeRegressor
# Creation of model

r_dt = DecisionTreeRegressor(random_state=0)

r_dt.fit(TOEFL,ADMISSION)
# Prediction

prediction5 = r_dt.predict(TOEFL)

prediction5
# Library

from sklearn.ensemble import RandomForestRegressor
# Creation of model

# Here, estimators mean how many decision tree will be created. 

rf_reg = RandomForestRegressor(n_estimators=10, random_state=0)

rf_reg.fit(TOEFL,ADMISSION)
# Prediction

prediction6 = rf_reg.predict(TOEFL)

prediction6
from sklearn.metrics import r2_score



print('Linear Regression R2 value:')

print(r2_score(admission, lr1.predict(toefl)))

print('**********************************************')

print('SVR R2 value:')

print(r2_score(admission_scaled, svr_reg.predict(admission_scaled)))

print('**********************************************')

print('Decision Tree R2 value:')

print(r2_score(ADMISSION, r_dt.predict(TOEFL)))

print('**********************************************')

print('Random Forest R2 value:')

print(r2_score(ADMISSION, rf_reg.predict(TOEFL)))