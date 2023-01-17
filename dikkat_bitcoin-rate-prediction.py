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
data=pd.read_csv('../input/Bitcoin.csv')
data.head()
#Here we don't need columns close_ratio, spread, and market. So, i am gonna drop that columns.

data.drop(columns=['close_ratio', 'market', 'spread','slug','symbol','name','ranknow','volume'], inplace = True)

data
#As we can see that we have dropped the columns that are not required.

data.shape

#We have 2039 rows and 5 columns in the data.
#Let's find min, count, max etc now to know more about the data.

data.describe()
data.dtypes
#As we can see that date is not in the date format. So, i will convert it into date datatype.

data['date'] = pd.to_datetime(data['date'])

data.dtypes
#date is successfully converted into date format
data.head()
#Also we don't need the columns high and low. So, i am gonna drop them too.

data.drop(columns=['high', 'low'], inplace = True)

data.head()

#linear regression

from sklearn.linear_model import LinearRegression
#splitting the data into train and test set

from sklearn.model_selection import train_test_split
#Let Set the date as index

data.set_index('date', inplace = True)
#import the matplot library

from matplotlib import pyplot as plt
x = data['open']

y = data['close']

plt.figure(figsize=(15,12))

plt.plot(x, color='red')

plt.plot(y, color = 'blue')

plt.show()

plt.xlabel('Open')

plt.ylabel('Close')

#plotting the graph between open and close attribute to see the relation between them
#Sorting the index 

data.sort_index(inplace=True)
X = data['open']

Y = data['close']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

X_test.tail()

#random_state-> same random splits
reg=LinearRegression()
x_train = np.array(X_train).reshape(-1,1)

x_test = np.array(X_test).reshape(-1,1)
reg.fit(x_train,y_train) 
reg.score(x_train, y_train)
plt.figure(figsize=(15,12))

plt.plot(y_train)

plt.plot(y_test)
