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
dataset=pd.read_csv('/kaggle/input/avocado-prices/avocado.csv')

dataset.columns

dataset=dataset.rename(columns={'Unnamed: 0':'index'})

dataset.head()

dataset=dataset.set_index('index')

dataset.head()
features=['Date','AveragePrice','Total Volume','Small Bags','Large Bags','XLarge Bags','type','year','region']

dataset=dataset[features]

dataset.head()
#checking on null values

dataset.isnull().sum()

#no Null values good to go !

#preprocessing date

dataset['Date']=dataset.Date.apply(lambda x:x.split('-')[1])
dataset.head(4)

#sorted the dates according to months 

dataset.groupby(['Date','year']).sum().sort_values(by="Total Volume",ascending=False).head(4).plot.bar()

#in the month of may the sales of avocado was the highest in 2016 followed by 2017 twice making to the top 4

dataset.groupby('region').sum().sort_values(by='Total Volume',ascending=False).head(4).plot.bar()

#The us region sells the highest avocados over the world with a huge difference 
dataset.head(5)
#checking the correlation between Average price and total volume sold 

dataset=dataset.rename(columns={'Total Volume':'volume'})

dataset.head(4)

table=pd.crosstab(dataset.AveragePrice,dataset.volume)

import scipy.stats

analysis=scipy.stats.chi2_contingency(table)

print(analysis)

#wow p value is hitting 99 percent which indicates a strong dependency

import matplotlib.pyplot as plt

plt.scatter(dataset.AveragePrice.head(1000),dataset.volume.head(1000),color="red")

#Analysis on US as it is the highest avocado seller 

X=dataset.loc[dataset.region=='TotalUS']

X.head()
sum=X['Small Bags'].sum()

print(sum)

sum1=np.log1p(sum)

print(sum1)
sum2=np.log1p(X['Large Bags'].sum())

print(sum2)
sum3=np.log1p(X['XLarge Bags'].sum())

print(sum3)
x_axis=['small bags','large bags','xlarge bags']

y_axis=[20.9,19.7,16.8]

plt.bar(x_axis,y_axis)

#The sales of the US are high in small bags compared to other two types of bags