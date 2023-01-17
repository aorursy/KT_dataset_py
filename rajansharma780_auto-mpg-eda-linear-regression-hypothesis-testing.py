# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import scipy.stats as stats



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#Load the csv file and make the data frame

auto_df = pd.read_csv('../input/auto-mpg.csv')
#display the data frame

auto_df
#display the shape of the data

print("The data frame has {} rows and {} columns".format(auto_df.shape[0],auto_df.shape[1]))
#display the data types of each column

auto_df.dtypes
#check null values are there or not

auto_df.apply(lambda x:sum(x.isnull()))
#calculate how many are '?' values in horsepower column

auto_df[auto_df['horsepower']=='?']
#5 point summary

auto_df.describe()
auto_df.replace('?',np.NAN,inplace=True)
auto_df['horsepower'] = auto_df['horsepower'].astype('float')
#check the data type of each column

auto_df.dtypes
#5 point summay

auto_df.describe()
#replace the Nan values of horsepower column with mean value of horsepower column

auto_df.replace(np.NaN,np.mean(auto_df['horsepower']),inplace=True)
#5 point summary of horsepower column

auto_df['horsepower'].describe()
#Now our objective is to predict the mpg that means mpg is our target variable. 

#so for that we have to check which independent variable has linear relationship with mpg(target variable)

#so for that we are using r value or pearon ceofficient

#if pearson coefficient is 0.7 in magnitude or greater that this we are considering there is linear relationship between independent variable and target variable

#if pearson coefficient is less than 0.7 in magnitude than we are considering there is not good linear relationship between independent variable and target variable



#so calculate pearson coefficient for each independent variable with target variable



print("the r value and p value for cylinders and mpg respectively is {}".format(stats.pearsonr(auto_df['cylinders'],auto_df['mpg'])))
for i in auto_df.columns:

    if i!='car name':

        print("the r and p value for "+i+" and mpg respectively is {}".format(stats.pearsonr(auto_df[i],auto_df['mpg'])))
#multivariate plot

sns.pairplot(data=auto_df)

plt.show()
#check how many cylinder cars are most

sns.countplot(auto_df['cylinders'])

plt.show()
#display most cars are of which model

sns.countplot(auto_df['model year'])

plt.show()
sns.jointplot(auto_df['model year'],auto_df['mpg'])

plt.show()
X = auto_df.iloc[:,1:8].values

Y = auto_df.iloc[:,0].values
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3,random_state=0)



regressor = LinearRegression()

regressor.fit(X_train,Y_train)
Y_pred = regressor.predict(X_test)

print(regressor.score(X_test,Y_test))