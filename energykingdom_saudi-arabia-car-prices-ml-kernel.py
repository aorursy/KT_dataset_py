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
import matplotlib.pyplot as plt 

import seaborn as sns
df = pd.read_csv('/kaggle/input/saudi-arabia-car-prices-machine-learning/carsclean.csv')
df.head(5)
df.shape #Data frame has 560 rows
df.dtypes #checking data types
df.isnull().sum()

# Checking nulls
print("precentage of nulls :",200/600)

# precentage of nulls
df.car_maker.unique() #checking unique values in car_maker column 
df.price.describe() #looking into price data
pricey = df[df.price == 100000]

pricey.count() #count of cars price = 100000
df.price.value_counts()  #count of prices
plt.figure(figsize= (20,6))

sns.boxplot(data = df , x = 'price' , y = 'car_maker', );

#price according to car maker box plot
plt.figure(figsize= (20,6))

sns.boxplot(data = df , x = 'year' , y = 'price', );

#Price According to year boxplot
df.hist(figsize=(15,5), color = "b") #ploting count per price
df.car_maker.value_counts().plot(kind = "bar", color = "g") #ploting count of cars per maker