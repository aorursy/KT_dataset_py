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
dataset=pd.read_csv('/kaggle/input/videogamesales/vgsales.csv')

dataset.head()
import seaborn as sn

import matplotlib.pyplot as plt

plt.figure(figsize=(14,6))

x=dataset.groupby('Genre')['Name'].count().sort_values(ascending=False).head(4).plot.bar()

#Looks like a pretty well balanced dataset with Action games selling a lot
plt.figure(figsize=(14,6))

x=dataset.groupby('Platform')['Rank'].count().sort_values(ascending=False).head(6).plot.line()

# playing games on famous consoles.
dataset.columns

dataset1=dataset.iloc[:,[6,7,8,9,10]]

dataset1.columns



plt.figure(figsize=(14,6))

sn.lineplot(data=dataset1.head(30))

dataset1.isnull().sum()

#The sales difference between different platforms.
dataset.head(4)

dataset.groupby('Year')['Global_Sales'].sum().sort_values(ascending=False).head(10).plot.bar()

#Highest global sales in the year 2008
plt.figure(figsize=(14,6))

dataset.groupby('Name')['Global_Sales'].sum().sort_values(ascending=False).head(6).plot.bar()

#Highest selling games 
plt.figure(figsize=(14,6))

sn.heatmap(data=dataset1.head(20))

#Analysis on 
dataset.head()
#Publisher analysis

plt.figure(figsize=(14,6))

dataset.groupby('Publisher')['Rank'].sum().sort_values(ascending=False).head(4).plot.bar()

#Results on the top manufacturers
plt.figure(figsize=(14,6))

dataset.groupby(['Publisher','Year'])['Global_Sales'].sum().sort_values(ascending=False).head(10).plot.bar()

#Best publisher and the best sales of some top years
#Multiple linear regression 

#predicting the global sales 

X=dataset.iloc[:,[6,7,8,9]]

Y=dataset.iloc[:,10]

from sklearn.linear_model import LinearRegression

plt.figure(figsize=(14,6))

sn.regplot(x=dataset['NA_Sales'],y=dataset['Global_Sales'])

#NA sales vs Global sales
dataset.head()
sn.regplot(x=dataset['EU_Sales'],y=dataset['Global_Sales'])

#Outlier to be taken in regards
sn.regplot(x=dataset['JP_Sales'],y=dataset['Global_Sales'])
sn.regplot(x=dataset['Other_Sales'],y=dataset['Global_Sales'])
from sklearn.model_selection import train_test_split

xtrain,xtest,ytrain,ytest=train_test_split(X,Y,test_size=0.2)

from sklearn.linear_model import LinearRegression

regressor=LinearRegression()

regressor.fit(xtrain,ytrain)

ypred=regressor.predict(xtest)

from sklearn.metrics import r2_score

print(r2_score(ytest,ypred)*100)



plt.figure(figsize=(14,6))

sn.lineplot(data=ypred)

#Analysis on ypred