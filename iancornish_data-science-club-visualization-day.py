# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
train.head()
# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt 
import seaborn as sns 
from bokeh.plotting import figure, output_file, show
plt.scatter('LotArea','SalePrice', data=train)
plt.scatter('LotArea','SalePrice', data=train)

plt.xlabel('Lot Area')
plt.ylabel('Sale Price')
plt.title('Sale Price by Lot Area')
plt.scatter('LotArea','SalePrice', data=train)

plt.xlabel('Lot Area')
plt.ylabel('Sale Price')
plt.title('Sale Price by Lot Area')

#feed the axis ranges in as ([xmin, xmax, ymin, ymax])
plt.axis([0,55000, 0, 500000])
plt.scatter('LotArea','SalePrice', c= 'OverallQual', data=train)

plt.xlabel('Lot Area')
plt.ylabel('Sale Price')
plt.title('Sale Price by Lot Area')

#feed the axis ranges in as ([xmin, xmax, ymin, ymax])
plt.axis([0,55000, 0, 500000])
#Creat a new dataframe with the variables that we want to look into
qualPrice = train[['OverallQual', 'SalePrice']]
#Create a boxplot of SalePrice grouped by the overall quality of the house
qualPrice.boxplot(by='OverallQual')
plt.show()
sns.pairplot(train, vars = ['SalePrice', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'GrLivArea', 'FullBath','HalfBath','BedroomAbvGr', 'TotalBsmtSF','1stFlrSF', '2ndFlrSF', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch','3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'YrSold'] )
plt.show()
sns.pairplot(train, vars = ['SalePrice', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'GrLivArea', 'FullBath','HalfBath','BedroomAbvGr', 'TotalBsmtSF','1stFlrSF', '2ndFlrSF', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch','3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'YrSold'], kind="reg" )
plt.show()
tooltips = [('Year-Built', '@YearBuilt'),
           ('Floors', '@HouseStyle'),
           ('Full-Bathrooms', '@FullBath'),
           ]
plot = figure( tooltips=tooltips,title="Sale Price by Lot Size")
plot.circle('LotArea', 'SalePrice', color="#42f450", source=train)
output_file('housing.html')
show(plot)