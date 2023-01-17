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
Head_Brain_Data = pd.read_csv("../input/headbrain/headbrain.csv") # reading the data set

Head_Brain_Data                                                   # view the data set

Head_Brain_Data.head()                                            # head of data set

Head_Brain_Data.tail()                                            # tail of data set
Head_Brain_Data.describe()                # description of data set
Head_Brain_Data.isnull().any()        # to check any null values


Head_Brain_Data.shape         # dimension of data set
Head_Brain_Data["Gender"].value_counts()      # levels of Gender
Head_Brain_Data["Age Range"].value_counts()          # levels of age range
Head_Brain_Data.keys()         # column names
y = Head_Brain_Data.iloc[:,3]           # getting target variable

y
x = Head_Brain_Data.iloc[:,2:3]     # independent variable

x
x.values       # converting into 2-D array to fit into linear regression model.
from sklearn.linear_model import LinearRegression # importing library for linear regression
model = LinearRegression()   # calling linear regression
model.fit(x,y)            # fitting the model
m= model.coef_                 # slope value

c= model.intercept_            # intercept value
y_pred = m*x + c                    # predicted values.

y_pred

y              # actual values
import matplotlib.pyplot as plt       # importing matplotlib
plt.scatter(x,y)                   # scatter plot for visualisation.

plt.plot(x,y_pred,c="red")
from sklearn.metrics import r2_score

print("R-squared :",r2_score(y,y_pred))