
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
train=pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/test.csv')
train.head()

train.describe(include='all')
print(train.shape)
print(test.shape)

#Getting a few details about the data sets

import matplotlib as plt
train.columns[train.isnull().any()]
#identifies the columns which has missing values
train._get_numeric_data()
#Now i will change all the numerical data missing values with its median value by median fitting model
new_median=train
new_median[new_median._get_numeric_data().columns].fillna(new_median[new_median._get_numeric_data().columns].median())
new_median.head()
correction=train[train.select_dtypes(exclude=['float64','int']).columns].mode()

#d=train.select_dtypes(exclude=['float64','int']).columns3
#correction.columns
for i in range(0,43):
    new_median[correction.columns[i]].fillna(correction[correction.columns[i]])
correction
#
#new_median
#new_median[train.select_dtypes(exclude=['float64','int']).columns].fillna(correction[d])
#new_median
#new_median["Alley"].fillna(correction[1])
correction[0:1:2]
#We will change the sale price values with their logarithmic value in order to reduce big values and scale
train['LogPrice']=np.log(train['SalePrice'])
train.head()
train['LogPrice'].describe()
import matplotlib.pyplot as plot
plot.hist(train.SalePrice)
plot.xlabel('SalePrice')
plot.ylabel('Frequency')
#Now we will check the skew of the saleprice and its log
print(train.SalePrice.skew())
print(train.LogPrice.skew())
#We notice that using logarithmic values reduces the skew a lot
train.corr().SalePrice
#Since from now on we are using LogPrice as main variable we are checking its correlation with other values
train.shape
nameval = train.select_dtypes(include='object').columns.values
for i in range (0,43):
    train.boxplot(column='SalePrice', by=nameval[i])
    plot.show()
numval=train._get_numeric_data().columns
for i in range (0,39):
    plot.scatter(train.SalePrice,train[numval[i]])
    plot.legend()
    plot.show()
    
    #this denotes the correlation between numerical variables and Sale price
#Since the highest correlation of the SalePrice is with GrLivArea For Building model We will use that reference data
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
X=train.iloc[:,46:47].values
Y=train.iloc[:,80].values
lin_reg.fit(X,Y)
plot.scatter(X,Y,color='red')
plot.plot(X,lin_reg.predict(X),color='blue')

#Calculating The R2 value to check validity
lin_reg.score(X,Y)
#We notice that linear plot is not that effective in this case
#We now try a polynomial regression fitting
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=3)
X_poly=poly_reg.fit_transform(X)
lin_reg_2=LinearRegression()
lin_reg_2.fit(X_poly,Y)
plot.plot(X,lin_reg_2.predict(poly_reg.fit_transform(X)))
plot.scatter(X,Y ,color='red')
lin_reg_2.score(X_poly,Y)
#lin_reg_2.predict(poly_reg.fit_transform(1710))
lin_reg.predict()
