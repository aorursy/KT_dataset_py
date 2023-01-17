# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib as plt

%matplotlib inline



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#This is a practice machine learning model that I've created for the first time and I am just trying out machine learning.

#It isnt a very accurate model but I wanted to get started after doing the micro courses. I ll continue learning and will make changes to this in the future
data_filepath = '../input/used-car-dataset-ford-and-mercedes/bmw.csv'

data = pd.read_csv(data_filepath)
data
data['transmission'].describe()
data.describe()
#Data Visulization

#To check the relationship between car price and year of manufacture

sns.regplot(x='year', y='price', data = data)
#To check the relationship between the price and the mileage

sns.regplot(x='mileage', y='price', data = data)
sns.barplot(x = 'transmission', y = 'price', data = data)
#To check the relationship between fuelType and price of the car

sns.barplot(x = 'fuelType', y = 'price', data = data)
#To check if we have any null values to take care of in the dataset

data.isnull().sum()
data.dtypes
#Handling categorical values of Transmission column

data['transmission'] = data['transmission'].str.replace('Automatic','1')

data['transmission'] = data['transmission'].str.replace('Semi-Auto','2')

data['transmission'] = data['transmission'].str.replace('Manual transmission','3')

data['transmission'] = data['transmission'].str.replace('Manual','4')
data['transmission'] = data['transmission'].astype(float)
#Handling categorical data in fuelType columns

data['fuelType'] = data['fuelType'].str.replace('Diesel','1')

data['fuelType'] = data['fuelType'].str.replace('Petrol','2')

data['fuelType'] = data['fuelType'].str.replace('Electric','3')

data['fuelType'] = data['fuelType'].str.replace('Diesel','4')

data['fuelType'] = data['fuelType'].str.replace('Hybrid','5')

data['fuelType'] = data['fuelType'].str.replace('Other','6')

data['fuelType'] = data['fuelType'].astype(float)

data.columns
#Splitting the data in training and testing data

from sklearn.model_selection import train_test_split



#Defining the target

y = data.price



#defining the features for the set

features = ['year', 'transmission', 'mileage', 'fuelType', 'tax', 'mpg', 'engineSize']

X = data[features]
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)
#Defining a model

from sklearn.linear_model import LinearRegression

bmw_model = LinearRegression()

bmw_model.fit(train_X, train_y)
price_pred = bmw_model.predict(val_X)

price_pred
from sklearn.metrics import mean_absolute_error

mean_absolute_error(val_y, price_pred)

#Hence, we get an average error of about $4444 in our prices which is not good and requires for fine tuning our model in order to get better results.