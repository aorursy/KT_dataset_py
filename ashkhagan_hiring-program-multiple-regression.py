# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns 

import matplotlib.pyplot as plt 

from sklearn import linear_model

import statsmodels.api as sm

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv("/kaggle/input/fcwrwrwa/Hiring - Sheet1 (1).csv")
df.columns
df.drop(['S.No.','Name'],axis=1)
plt.scatter(df['Experience'], df['salary'], color='red')

plt.title('Experience vs Salary', fontsize=14)

plt.xlabel('Experience', fontsize=14)

plt.ylabel('Salary', fontsize=14)

plt.grid(True)

plt.show()
plt.scatter(df['Interviewscore'], df['salary'], color='red')

plt.title('InterviewScore vs salary', fontsize=14)

plt.xlabel('Interviewscore', fontsize=14)

plt.ylabel('Salary', fontsize=14)

plt.grid(True)

plt.show()
plt.scatter(df['Writtenscore'], df['salary'], color='red')

plt.title('Writtenscore vs salary', fontsize=14)

plt.xlabel('Writtenscore', fontsize=14)

plt.ylabel('Salary', fontsize=14)

plt.grid(True)

plt.show()
df.columns
X = df[['Experience','Writtenscore','Interviewscore']] # here we have 2 variables for multiple regression. If you just want to use one variable for simple linear regression, then use X = df['Interest_Rate'] for example.Alternatively, you may add additional variables within the brackets

Y = df['salary']

 

# with sklearn

regr = linear_model.LinearRegression()

regr.fit(X, Y)



print('Intercept: \n', regr.intercept_)

print('Coefficients: \n', regr.coef_)



# prediction with sklearn

Experience=5

Interviewscore=10

Writtenscore=9

print ('Predicted Salary: \n', regr.predict([[Experience,Writtenscore,Interviewscore]]))



# with statsmodels

X = sm.add_constant(X) # adding a constant

 

model = sm.OLS(Y, X).fit()

predictions = model.predict(X) 

 

print_model = model.summary()

print(print_model)
# prediction with sklearn

Experience=8

Interviewscore=6

Writtenscore=7

print ('Predicted Salary: \n', regr.predict([[Experience,Writtenscore,Interviewscore]]))

# with statsmodels

X = sm.add_constant(X) # adding a constant

 

model = sm.OLS(Y, X).fit()

predictions = model.predict(X) 

 

print_model = model.summary()

print(print_model)
