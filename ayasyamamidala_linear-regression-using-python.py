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
import pandas as pd

dx=pd.read_csv("../input/salary-data-simple-linear-regression/Salary_Data.csv") #It reads the comma seperated values from local files.

dx.keys()





from sklearn.linear_model import LinearRegression #including a library for linear regression

machinebrain=LinearRegression() #independent variable must be in 2D format

x=dx.iloc[:,0:1].values #converts x into 2D format

y=dx.iloc[:,1].values #converts y into 1D format
machinebrain.fit(x,y) #This function fits x&y on the graph to calculate LinearRegression

m=machinebrain.coef_ #This function is used to fit best slope for the dataset automaticaly

c=machinebrain.intercept_ #This function is used to fit best intercept for the dataset automatically

y_predict=m*x+c #This is the straight line equation which plots the slope & intercept values.
import matplotlib.pyplot as plt #It is a library imported for data visualisation

plt.scatter(x,y) #plots scatter points between x&y

plt.plot(x,y_predict,c="green",marker="*")

plt.show()
h = 5.5 #The given value of 5.5 yrs of experience for the given amount of salary

w = machinebrain.predict([[h]])

plt.scatter(x,y)

plt.plot(x,y_predict, c="indigo")

plt.scatter([h],w,c="orange")

plt.xlabel("Experience")

plt.ylabel("Salary")

plt.show()