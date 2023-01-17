# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import linear_model

import statsmodels.api as sm

import matplotlib.pyplot as plt

from sklearn import linear_model



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv("../input/promotional-mix/Data_Regression.csv")
X = df[['Calls_Spend','Conference_Spend','Email_Spend','Display_Spend']]

Y = df['Total_Revenue']
# with sklearn

regr = linear_model.LinearRegression()

regr.fit(X, Y)
print('Intercept: \n', regr.intercept_)

print('Coefficients: \n', regr.coef_)
New_Calls_Spend= 43064406



New_Conference_Spend = 2419652





New_Email_Spend=1501195.038



New_Display_Spend=2000078



Total_Spend=New_Calls_Spend+New_Conference_Spend+New_Email_Spend+New_Display_Spend

print ('Predicted Revenue: \n', regr.predict([[New_Calls_Spend,New_Conference_Spend,New_Email_Spend,New_Display_Spend ]]))

print ('Total Spend:\n',Total_Spend)

print ('Revenue as a percentage of spend: \n',regr.predict([[New_Calls_Spend,New_Conference_Spend,New_Email_Spend,New_Display_Spend ]])/Total_Spend)
# with statsmodels

X = sm.add_constant(X) # adding a constant
model = sm.OLS(Y, X).fit()

predictions = model.predict(X) 
print_model = model.summary()

print(print_model)