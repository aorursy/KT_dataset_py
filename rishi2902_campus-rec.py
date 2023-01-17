# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session



# 1)

import pandas as pd

import numpy as np



df = pd.read_csv('/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')

print(df.columns)

print(df.head())



print("\n\nLR model for ssc_p and hsc_p with output variable mba_p")

# Separating the data into input and output variables

X = df.iloc[:, [2, 4]].values

Y = df.iloc[:, 12].values.reshape(-1, 1)



# Splitting the datset into input and the output variables

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=5)



# fitting the linear regression model

from sklearn.linear_model import LinearRegression



model = LinearRegression()

model.fit(x_train, y_train)



y_pred = model.predict(x_test)



opt = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})

print(opt)



print('\nRegression coefficients :')

print('Slope :', model.coef_)

print('y_intercept:', model.intercept_)

print('\n==============================================================\n\n')

# 2)

print("LR model for ssc_p and degree_p with output variable mba_p")

# Separating the data into input and output variables

X = df.iloc[:, [2, 7]].values

Y = df.iloc[:, 12].values.reshape(-1, 1)



# Splitting the datset into input and the output variables

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=5)



# fitting the linear regression model

from sklearn.linear_model import LinearRegression



model = LinearRegression()

model.fit(x_train, y_train)



y_pred = model.predict(x_test)



opt = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})

print(opt)

from sklearn.metrics import r2_score

print('\nR-squared metric :', r2_score(y_test, y_pred))

print('\n-----------------------------------------------------------\n')

print("LR model for hsc_p and degree_p with output variable mba_p")

# Separating the data into input and output variables

X = df.iloc[:, [4, 7]].values

Y = df.iloc[:, 12].values.reshape(-1, 1)



# Splitting the datset into input and the output variables

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=5)



# fitting the linear regression model

from sklearn.linear_model import LinearRegression



model = LinearRegression()

model.fit(x_train, y_train)



y_pred = model.predict(x_test)



opt = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})

print(opt)

print('R-sqrared metric : ', r2_score(y_test, y_pred))



print('\n\nAs we see that r2 score for first one grater than the second one\n'

      'therefore model with ssc_p and degree_p is better than that of hsc_p and degree_p')



# 3)

print('\n==============================================================\n\n')

print("LR model for ssc_p, hsc_p and degree_p with output variable mba_p")

# Separating the data into input and output variables

X = df.iloc[:, [2, 4, 7]].values

Y = df.iloc[:, 12].values.reshape(-1, 1)



# Splitting the datset into input and the output variables

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=1001)



# fitting the linear regression model

from sklearn.linear_model import LinearRegression



model = LinearRegression()

model.fit(x_train, y_train)

print('\nRegression coefficients :')

print('Slope :', model.coef_)

print('y_intercept:', model.intercept_)

y_pred = model.predict(x_test)

f_opt = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})

print(f_opt)
