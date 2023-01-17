# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder, StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures

from sklearn.svm import SVR 

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import r2_score

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/maaslar.csv", sep=",")

df.columns = ['Title', 'Education Level', 'Salary']

df.head()
df.describe()
df.info()
educationLevel = df.iloc[:,1:2]

salary = df.iloc[:,2:3]
sc1 = StandardScaler()

scaledEducationLevel = sc1.fit_transform(educationLevel)

sc2 = StandardScaler()

scaledSalary = sc2.fit_transform(salary)
modelSVR = SVR(kernel='rbf')

modelSVR.fit(scaledEducationLevel, scaledSalary)

y_pred = modelSVR.predict(scaledEducationLevel)

y_pred
plt.scatter(scaledEducationLevel, scaledSalary, color = "red")

plt.plot(scaledEducationLevel, y_pred, color = "blue")      
modelDT = DecisionTreeRegressor(random_state = 0)

modelDT.fit(educationLevel, salary)
plt.scatter(educationLevel, salary, color = "red")

plt.plot(educationLevel, modelDT.predict(educationLevel), color="blue")

plt.show()
A = educationLevel + .5

B = educationLevel - .4

plt.scatter(educationLevel, salary, color = "red")

plt.plot(educationLevel, modelDT.predict(educationLevel), color="blue")

plt.plot(educationLevel, modelDT.predict(A), color ="green") 

plt.plot(educationLevel, modelDT.predict(B), color = "orange") 

plt.show()
print(modelDT.predict(educationLevel))
print(modelDT.predict(np.array([[10],[11],[20],[50]])))
modelRF = RandomForestRegressor(n_estimators = 10, random_state = 0)

modelRF.fit(educationLevel, salary)
y_pred = model.predict(educationLevel)

plt.scatter(educationLevel, salary, color = "red")

plt.plot(educationLevel, y_pred, color ="blue")

plt.show()
plt.scatter(educationLevel, salary, color = "red")

plt.plot(educationLevel, y_pred, color ="blue")

plt.plot(educationLevel, modelRF.predict(A), color ="green") 

plt.plot(educationLevel, modelRF.predict(B), color = "orange")

plt.show()

print(modelRF.predict(np.array([[1],[5],[20],[50]])))
print(r2_score(salary, modelRF.predict(educationLevel)))
print(r2_score(salary,modelDT.predict(educationLevel)))
print(r2_score(scaledSalary,modelSVR.predict(scaledEducationLevel)))