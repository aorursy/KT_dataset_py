# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from sklearn import linear_model

import matplotlib.pyplot as plt  

from sklearn.linear_model import LinearRegression

import os

print(os.listdir("../input"))





# Any results you write to the current directory are saved as output.
#Dataset

file = pd.read_csv("../input/heart.csv")

file
#Visualize Data #1 (age vs. target)

age = file['age']

heart_disease = file['target']

plt.scatter(age, heart_disease, edgecolors='r')

plt.xlabel('Age')

plt.ylabel('Presence of Heart Disease')

plt.title('Presence of Heart Disease by Age')

plt.show()
#Percentage of 58 year olds with Heart Disease

counter = 0

x = file.age

y = file.target

for i in range(0,303):

    if (x[i] == 58):

        if (y[i] == 1):

            counter = counter + 1

print(round(counter/303*100,2), "percent of people who have Heart Disease are 58 Years Old in this dataset.")
#Visualize Data #2 (chol vs. target)

cholesterol = file['chol']

heart_disease = file['target']

plt.scatter(cholesterol, heart_disease, edgecolors='b')

plt.xlabel('Cholesterol (mg/dl)')

plt.ylabel('Presence of Heart Disease')

plt.title('Presence of Heart Disease by Cholesterol')

plt.show()
#Percentage of people with Heart Disease who have more than 300 mg/dl levels of Cholesterol

counter = 0

x = file.chol

y = file.target

for i in range(0,303):

    if (x[i] > 300):

        if (y[i] == 1):

            counter = counter + 1

print(round(counter/303*100,2), "percent of people with Heart Disease have more than 300 mg/dl levels of Cholesterol in this dataset.")
#Linear Regression Plot of age vs. target (Machine Learning Algorithm)

x = file['age'].values[:,np.newaxis]

y = file['target'].values



regressor = LinearRegression()

regressor.fit(x,y)

plt.scatter(x, y,color='g')

plt.plot(x, regressor.predict(x),color='k')



plt.show()
#Linear Regression Correlation Coefficients (Machine Learning Algorithm)

lines = linear_model.LinearRegression()

lines.fit(file[['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']],file.target)

lines.coef_
