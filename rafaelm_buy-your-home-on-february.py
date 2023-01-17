# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np



from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#As the data has a date column let's parse the dates for later analisys

df = pd.read_csv('../input/kc_house_data.csv', parse_dates = ['date'])

df.head()

#Kind of data we are working

df.info()
#Find null values

df.isnull().sum()
#Extract from 'date of sale' and create two new columns with month an year to find relations with price

df['month'] = df['date'].dt.month

df['year'] = df['date'].dt.year

df.head()
#Correlation

corr = df.corr()



#Listed correlation with the price. 23 is the number of columns of the df

corr.nlargest(23, 'price')['price']



#Mybe there is not correlation between dates and price but the prices can be stationary

#I will also check if the prices had increase from 2014 to 2015



#Price increase between 2014 adn 2015

priceYear =  df['price'].groupby(df['year']).mean()

priceYear.plot(kind = 'bar')



#% value of the icrease

list_priceYear = list(priceYear)

priceIncrease = ((list_priceYear[0]/list_priceYear[1])-1)*(-100)

print ('Form 2014 to 2015 there is a price increase in % of: ', priceIncrease)
#Find if the prices are stationary between months

priceMonth = df['price'].groupby(df['month']).mean()

priceMonth.plot(kind = 'line')

#The difference in the average price between February and May

print('The average price diference in $ buying a house in Feb or in May is: ', priceMonth.max()-priceMonth.min())
#Create the data to train.

y = df['price']

df = df.drop(['price', 'id', 'date'], axis = 1)

x_train,x_test,y_train,y_test=train_test_split(df,y,train_size=0.8,random_state=42)



#Linnear regression

reg=LinearRegression()

reg.fit(x_train,y_train)

reg.score(x_test,y_test)
