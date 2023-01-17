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
#regression is a form of predictive modelling technique which investigates the relationship between dependent and independent variable.

#linear regression is a type of regression which predeicts a dependent value based on independent value.

import pandas as pd #imports the pandas library and names it 'pd'

from sklearn.linear_model import LinearRegression #imports linearregression modules from sklearn library

df=pd.read_csv("../input/random-linear-regression/test.csv") #it reads the csv(comma seperated values) from local files

mb=LinearRegression() #calling the module to the variable

x=df.iloc[:,0:1] #converting the independent variable to 1d-array

y=df.iloc[:,1] # converting the dependent variable to 2d-array

mb.fit(x,y) #trains or fits the information in the mb





m=mb.coef_ #creating and assigning a coefficient to m

c=mb.intercept_ #creating and assigning a intercept

y_p=m*x+c #the straight line equation  which plots slopes and intercept values

y_p
#to plot values

import matplotlib.pyplot as p #importing a library for ploting values

p.scatter(x,y) #to create scatterplots of the values 

p.plot(x,y_p, c="yellow")

p.show() #to display the graph which shows the best fit line .The red line shows the best line which crosses most of the dots.








