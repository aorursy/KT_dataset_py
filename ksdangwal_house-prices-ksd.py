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
train_data = pd.read_csv("../input/train.csv")
train_data.head
train_data.describe()
train_data.head()
train_data.tail()
import seaborn as sns
train_data.groupby(['MSZoning']).mean()['SalePrice'].reset_index()
zone_sale_data = train_data.groupby(['MSZoning']).mean()['SalePrice'].reset_index()
sns.barplot(x='MSZoning', y='SalePrice', hue='MSZoning', data=zone_sale_data)
sns.scatterplot(x='MSZoning', y='SalePrice', hue='SalePrice', data=zone_sale_data)
year_sale_price = train_data[['YearBuilt', 'SalePrice']]
year_sale_price.head()
sns.boxplot(x='SalePrice', y='YearBuilt', data=year_sale_price)
train_data.groupby(['YearBuilt']).mean()['SalePrice'].head().reset_index()
mean_year_sale = train_data.groupby(['YearBuilt']).mean()['SalePrice'].reset_index()
sns.scatterplot(x='YearBuilt', y='SalePrice', hue='YearBuilt', data=mean_year_sale)
houseStyle_zone_sale = train_data.groupby(['HouseStyle', 'MSZoning']).mean()['SalePrice'].reset_index()
houseStyle_zone_sale
sns.barplot(x='HouseStyle', y='SalePrice', data=houseStyle_zone_sale)
sns.barplot(x='MSZoning', y='SalePrice', hue='HouseStyle', data=houseStyle_zone_sale)
def getNewHouseStyle(x) :
    return x + ' Apartment';

train_data['NewHouseStyle'] = train_data['HouseStyle'].map(lambda x : getNewHouseStyle(x))

train_data.head()
binning_home_style = train_data.groupby(['HouseStyle', 'NewHouseStyle']).mean()['SalePrice'].reset_index() 
binning_home_style
sns.barplot(x='HouseStyle', y='SalePrice', hue='NewHouseStyle', data=binning_home_style)