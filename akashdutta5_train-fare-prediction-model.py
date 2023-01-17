# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Importing Datadet in notebook

df = pd.read_csv('../input/renfe.csv')

df.head(5)
#Checking for null values

df.isnull().sum()
#Finding mean of price to be filled in null places

mean = df.price.mean()

mean
#Filling null values of price column

df.price.fillna(63.3855 , inplace = True)

df.isnull().sum()
# Filling train class coloumn`s null vales with Turista as it was the most common

df.train_class.fillna("Turista" , inplace = True)

df.isnull().sum()
#Filling fare coloumn`s null vales with Promo as it was the most common

df.fare.fillna("Promo" , inplace = True)

df.isnull().sum()
#Categorical chart between price and train class

import seaborn as sns

import matplotlib.pyplot as plt



sns.catplot(x='price' , y= 'train_class' , data = df)

plt.show()
# Categorical graph between price and train type

sns.catplot(x='price' , y= 'train_type' , data = df)

plt.show()
#Graph to show the distribution of price

sns.distplot(df['price'] , bins = 9 , hist = True)

plt.show()
# Deleting useless columns 

del df['insert_date']

del df['start_date']

del df['end_date']

del df ['Unnamed: 0']

df.info()
#Label encoding columns to make model train on it

from sklearn.preprocessing import LabelEncoder

label = LabelEncoder()

df['origin'] = label.fit_transform(df['origin'])

df['destination'] = label.fit_transform(df['destination'])

df['train_type'] = label.fit_transform(df['train_type'])

df['train_class'] = label.fit_transform(df['train_class'])

df['fare'] = label.fit_transform(df['fare'])

df.info()
#Splitting data into train and test

from sklearn.model_selection import train_test_split

train , test = train_test_split(df , test_size = 0.2 , random_state = 1)

#Further splitting

def data_splitting(df):

    x=df.drop(['price'], axis=1)

    y=df['price']

    return x, y

x_train , y_train = data_splitting(train)

x_test , y_test = data_splitting(test)

#Applying Linear regression model

from sklearn.linear_model import LinearRegression

log = LinearRegression()

log.fit(x_train , y_train)
log_train = log.score(x_train , y_train)

log_test = log.score(x_test , y_test)



print("Training score :" , log_train)

print("Testing score :" , log_test)

#Applying Random FOrest Regressor

from sklearn.ensemble import RandomForestRegressor

reg = RandomForestRegressor()

reg.fit(x_train , y_train)
reg_train = reg.score(x_train , y_train)

reg_test = reg.score(x_test , y_test)



print("Training Score :" , reg_train)

print("Testing Score :" , reg_test)