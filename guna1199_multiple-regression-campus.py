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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import random
campus_data = pd.read_csv('/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')

x1 = campus_data.iloc[:,[2,4]].values

x2 = campus_data.iloc[:,[4,7]].values

x3 = campus_data.iloc[:,[2,4,7]].values

y = campus_data.iloc[:,12].values

print(x1[0:5])
print(x2[0:5])
print(x3[0:5])
print(y[0:5])
from sklearn.model_selection import train_test_split

x1_train, x1_test, y_train, y_test = train_test_split(x1, y, test_size = 0.2,random_state = 0)

x2_train, x2_test, y_train, y_test = train_test_split(x2, y, test_size = 0.2,random_state = 0)

x3_train, x3_test, y_train, y_test = train_test_split(x3, y, test_size = 0.2,random_state = 0)
from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score

lr = LinearRegression()



lr.fit(x1_train,y_train)



y_pred1 = lr.predict(x1_test)



df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred1.flatten()})

df

#print((np.concatenate((y_test.reshape(43,1),  y_pred1.reshape(43,1)),1)))

print(lr.intercept_)

print(lr.coef_)
r2_score(y_test, y_pred1)

from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score

lr2 = LinearRegression()

lr2.fit(x2_train,y_train)

y_pred2 = lr2.predict(x2_test)

df2 = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred2.flatten()})

df2
print(lr2.intercept_)

print(lr2.coef_)
r2_score(y_test, y_pred2)
from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score

lr3 = LinearRegression()

lr3.fit(x3_train,y_train)

y_pred3 = lr3.predict(x3_test)

df3 = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred3.flatten()})

df3

print(lr3.intercept_)

print(lr3.coef_)
r2_score(y_test, y_pred3)
