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
import os
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
os.getcwd()

import pandas as pd
avo = pd.read_csv('../input/avocado.csv')
sns.pairplot(avo, x_vars='AveragePrice',y_vars='Total Volume')
#Pair plot is used to understand 
#or to explain a relationship between two variables 
#sns.pairplot(avo)

avo.head()
#How many rows and columns are there
avo.shape
#What is the average Purchase Price of Avocado in last 4 years
avo['AveragePrice'].mean()
#What is the minimum Price of Avocado in last 4 years
avo['AveragePrice'].min()
#What is the maximum Price of Avocado in last 4 years
avo['AveragePrice'].max()
#Find maximum no of Avacados are sold in last 4 years and which year it is?
avo['Total Volume'].max()

#How many people made the purchase of Organic and Conventional in last 4 years
avo['type'].value_counts()
#Is there a correlation between Total bags and Total volume
avo[['Total Volume','Total Bags']].corr()
sns.heatmap(avo[['Total Volume','Total Bags']].corr())
#'What was the average volume of Avocado per year? (2015-2018) ?')
avo.groupby('year')['Total Volume'].mean()
#Find Avg Price of Avocado in the different region in last 4 years
avo.groupby('region')['AveragePrice'].mean()
#mean of Average Price in last 4 years
avo['AveragePrice'].mean()
#Which year Avacodo costed more than average in among the diff PLU(Price look-up codes)

avo[avo['AveragePrice'] > avo['AveragePrice'].mean()][['AveragePrice','4046','4225','4770']].sort_values(by = ['AveragePrice'])
#Find Average price of Organic Avocado in last years of which regions?
#Subset of data from complete dataset 
organic = avo[avo['type'] == 'organic']
organic

#Find the average Organic Avacado cost in last 4 years
organic.groupby('year')['AveragePrice'].mean()

#Find the list of average Organic Avacado cost in different regions in last 4 years
organic.groupby('region')['AveragePrice'].mean().sort_values(ascending=False)
#Find sales by regions and later build it by year
avo.groupby('region')['Total Volume'].mean().sort_values()

#Find each year in last 4 years how many types of avacado got sold.
avo.groupby('year')['type'].value_counts()
#Find the total no of sales in each year of last 4 years
avo.groupby('year')['Total Volume'].count()
#Data avialable for each year..in the dataset for last 4 years(size of the dataframe)
avo.groupby('year')['Total Volume'].size()
#Correlation:Correlation is a statistical measure that indicates the extent to which two or more variables fluctuate together. 
avo[['4046','Small Bags','Total Volume','Total Bags']].corr()

sns.heatmap(avo[['4046','Small Bags','Total Volume','Total Bags']].corr())
sns.heatmap(avo[['4225','Large Bags','Total Volume','Total Bags']].corr())
sns.heatmap(avo[['4770','XLarge Bags','Total Volume','Total Bags']].corr())
sns.heatmap(avo[['4046','4770','4225','Small Bags','Large Bags','XLarge Bags','Total Volume','Total Bags']].corr())
#Average Sales of Avacados by month 
avo['Month'] = avo['Date'].apply(lambda date:pd.Period(date, freq='M'))
avo.head(2)
avg_monthly_sales = avo.groupby(avo['Month'])['Total Volume'].mean()
avg_monthly_sales
sns.distplot(avg_monthly_sales,bins=10, kde=False)

#Average Sales of Avacados by Quater of each year in last 4 years
avo['Quater'] = avo['Date'].apply(lambda date:pd.Period(date, freq='Q'))
avo.head(6)
avg_Q_sales = avo.groupby(avo['Quater'])['Total Volume'].mean()
avg_Q_sales
#Distribution plot showing the Average price in four years(Avg Price of Avacado ranged between $1.0 and $1.5)
sns.distplot(avo['AveragePrice'],bins=10, kde=False)
#Avg Price of Avacados greater than the Average Price of Avacados in last 4 years
# Looks like 2016 and 2017 first quater the the price was greater than average
sns.jointplot(x='AveragePrice', y='year', data=avo[avo['AveragePrice'] > avo['AveragePrice'].mean()], kind='hex', 
              gridsize=20)
#Average sales of PLU 4225 by region.
#avo.groupby('region')['4225'].mean()
sns.distplot(avo.groupby('region')['4225'].mean(),bins=10, kde=False)
#Average sales of PLU 4770 by region.
avo.groupby('region')['4770'].mean()
sns.distplot(avo.groupby('region')['4770'].mean(),bins=10, kde=False)
#Average sales of PLU 4026 by region.
#avo.groupby('region')['4046'].mean()
## Looks like PLU 4046 and PLU 4225 are preffered same across All S regions.
sns.distplot(avo.groupby('region')['4046'].mean(),bins=10, kde=False)



