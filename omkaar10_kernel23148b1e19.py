# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as mp

from pandas import DataFrame as df

from sklearn.linear_model import LinearRegression





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



#loading data

movies=pd.read_csv('/kaggle/input/movie-sale/Movie_Revenue_Orginal.csv')
movies.head()
movies.columns
#Cleaning Data

#getting rid of $ in both columns

movies['Production Budget ($)']=movies['Production Budget ($)'].str.replace('$','')

movies['Worldwide Gross ($)']=movies['Worldwide Gross ($)'].str.replace('$','')



#getting rid of , in both columns

movies['Production Budget ($)']=movies['Production Budget ($)'].str.replace(',','')

movies['Worldwide Gross ($)']=movies['Worldwide Gross ($)'].str.replace(',','')



#getting rid of extra space in both columns

movies['Production Budget ($)']=movies['Production Budget ($)'].str.lstrip()

movies['Worldwide Gross ($)']=movies['Worldwide Gross ($)'].str.lstrip()



#getting rid of extra space in both columns

movies['Production Budget ($)']=movies['Production Budget ($)'].str.rstrip()

movies['Worldwide Gross ($)']=movies['Worldwide Gross ($)'].str.rstrip()

#converting string to float type



movies['Production Budget ($)']=movies['Production Budget ($)'].astype(float)

movies['Worldwide Gross ($)']=movies['Worldwide Gross ($)'].astype(float)

x=movies['Production Budget ($)']

y=movies['Worldwide Gross ($)']
#plotting a scatter plot



mp.scatter(x,y)

mp.xlabel('Production Budget ($)')

mp.ylabel('Worldwide Gross ($)')

mp.show()
regressionObject=LinearRegression()
x=x.values.reshape(-1,1)

y=y.values.reshape(-1,1)

regressionObject.fit(x,y)
#plotting a regression line

mp.scatter(x,y)

mp.xlabel('Production Budget ($)')

mp.ylabel('Worldwide Gross ($)')

mp.plot(x,regressionObject.predict(x),color='black')

mp.show()
regressionObject.coef_
regressionObject.intercept_
#creating a dataframe



data=[[200000000]]

budget=df(data,columns=['Estimated_Budget'])

budget
#hypothesis which predicts the Worldwide gross



regressionObject.coef_[0][0] * budget.loc[0]['Estimated_Budget'] + regressionObject.intercept_[0]
regressionObject.score(x,y)