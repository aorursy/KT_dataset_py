# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#import pandas library and load my dataset to a dataframe df

import pandas as pd

df = pd.read_csv("../input/usedcarsales.csv")
#check top five rows



df['FuelType'] = df['FuelType'].map({'Diesel': 1, 'Petrol': 0, 'CNG': -1})

df.loc[:,:].sort_values(by=['Price'], ascending =False)

df['Price'].describe()

#rename first column

df = df.rename(columns={'Unnamed: 0' : 'index_num'})

df.head(2)
#check if the index_num is unique

df['index_num'].is_unique
#import libraries to plot

import matplotlib.pyplot as plt

%matplotlib inline

 

#plt.scatter(df['Price'],df['Weight'])

plt.scatter(df['Price'],df['Age'])

#plt.scatter(df['Price'],df['KM'])

#plt.scatter(df['Automatic'],df['Price'])

#plt.scatter(df['Price'],df['FuelType'])

#plt.scatter(df['HP'],df['Price'])

#plt.scatter(df['CC'],df['Price'])



#Assign X,y for our prediction model

X = df[['KM','Age','Weight','Automatic','FuelType','HP','CC']]

#X = df[['Age','Weight']]

y = df['Price']
#Load Train_Test_Split Model

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
X_train
y_train
y_test
#Fit the model for LinearRegression 

from sklearn.linear_model import LinearRegression

clf = LinearRegression()

clf.fit(X_train, y_train)
X_test
#predict X_test for prices

clf.predict(X_test)
#check predicted values against actual values

clf.score(X_test, y_test)
import seaborn as sns

sns.distplot(df['Price'])

print("Skewness: %f" % df['Price'].skew())

print("Kurtosis: %f" % df['Price'].kurt())

data = pd.concat([df['Price'],df['Age']], axis=1)

f, ax = plt.subplots(figsize=(16,8))

fig = sns.boxplot(x = df['Age'], y=df['Price'], data=data)

fig.axis(ymin=0, ymax=32500)

plt.xticks(rotation=90)
#Plot correlation heatmap

corrmat = df.corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, vmax=.8, square=True);
#find missing values in dataset

total = df.isnull().sum().sort_values(ascending=False)

percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(20)