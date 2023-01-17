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
# Importing the raw Avocado datasets

df = pd.read_csv('/kaggle/input/avocadoprices/avocado.csv')

df2 = pd.read_csv('/kaggle/input/avocado-plu/avocadoPLU.csv')

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from scipy import stats

from datetime import datetime

from scipy.stats import ttest_ind
#This will allow us to visualize some of the raw data and delete some columns that take up too much noise....



del df['4046 pct..1']

del df['4770 pct..1']

del df['4225 pct..1']

del df['4046 pct.']

del df['4770 pct.']

del df['4225 pct.']

df.head()
df2.head()
print(df.shape)

print(df2.shape)
#Creating a boxplot of the Average Price for Non-Organic Avocado's from 2015-2018 per City

sns.boxplot(x='region', y='AveragePrice',data=df[df.type=='Non-Organic']).set_title("Non-Organic (2015-2018) per city",fontsize=40)



plt.ylabel('Average Price', fontsize = 30)

plt.xlabel('Region', fontsize = 30)

plt.style.use('fivethirtyeight')

plt.xticks(rotation=90)







#creating a reference of the above boxplot so the size may be adjusted 

fig=plt.gcf()

fig.set_size_inches(20.7,9.27)

#Creating a boxplot of the Average Price for Non-Organic Avocado's from 2015-2018 per City

sns.boxplot(x='region', y='AveragePrice',data=df[df.type=='organic']).set_title("Organic (2015-2018) per city",fontsize=40)

plt.ylabel('Average Price', fontsize=30)

plt.xlabel('Region' ,fontsize=30)

plt.style.use('fivethirtyeight')

plt.xticks(rotation=90)



#creating a reference of the above boxplot so the size may be adjusted 

fig=plt.gcf()

fig.set_size_inches(20.7,9.27)
df['year'] =pd.to_datetime(df.Date).dt.year

df['months'] = pd.to_datetime(df.Date).dt.month


df.head()
sns.violinplot(x='type', y='AveragePrice',data=df, fontsize = 20)



plt.ylabel('Average Price' ,fontsize=30)

plt.xlabel('Type' ,fontsize=30)

plt.yticks(np.arange(0.5, 4.0, 0.30))

fig=plt.gcf()

fig.set_size_inches(25.7,10.27)

plt.figure(figsize=(12,10))



df[df.year==2015].groupby('months').AveragePrice.mean().plot()

df[df.year==2016].groupby('months').AveragePrice.mean().plot()

df[df.year==2017].groupby('months').AveragePrice.mean().plot()

df[df.year==2018].groupby('months').AveragePrice.mean().plot()

plt.legend([2015, 2016, 2017, 2018])

plt.ylabel('Average Price',fontsize=30)

plt.xlabel('Month',fontsize=30)

plt.title('Mean of the average price per month (Organic & Non-Organic)',fontsize=40)



fig=plt.gcf()

fig.set_size_inches(30.7,15.27)

plt.show()

plt.style.use('fivethirtyeight')

df[(df.year==2015) & (df.type=='Non-Organic' ) ].groupby('months').AveragePrice.mean().plot()

df[(df.year==2016) & (df.type=='Non-Organic' ) ].groupby('months').AveragePrice.mean().plot()

df[(df.year==2017) & (df.type=='Non-Organic' ) ].groupby('months').AveragePrice.mean().plot()

df[(df.year==2018) & (df.type=='Non-Organic' ) ].groupby('months').AveragePrice.mean().plot()



plt.legend([2015, 2016, 2017, 2018])



plt.xticks(np.arange(2, 13, 2))

plt.yticks(np.arange(0.8, 2.0, 0.15))

plt.ylabel('Average Price',fontsize=30)

plt.xlabel('Month',fontsize=30)

plt.title('Mean of the average price for Non-Organic per month',fontsize=40)



fig=plt.gcf()

fig.set_size_inches(30.7,15.27)

plt.show()
plt.style.use('fivethirtyeight')

df[(df.year==2015) & (df.type=='organic' ) ].groupby('months').AveragePrice.mean().plot()

df[(df.year==2016) & (df.type=='organic' )].groupby('months').AveragePrice.mean().plot()

df[(df.year==2017) & (df.type=='organic' ) ].groupby('months').AveragePrice.mean().plot()

df[(df.year==2018) & (df.type=='organic' ) ].groupby('months').AveragePrice.mean().plot()

plt.legend([2015, 2016, 2017, 2018])



plt.yticks(np.arange(1.2, 2.8, 0.10))

plt.ylabel('Average Price',fontsize=30)

plt.xlabel('Month',fontsize=30)

plt.title('Mean of the average price for Organic per month',fontsize=40)



fig=plt.gcf()

fig.set_size_inches(30.7,15.27)

plt.show()
plt.hist(df[df.type=='Non-Organic'].AveragePrice ,label='Non-Organic', alpha = 0.25)

plt.hist(df[df.type=='organic'].AveragePrice,label='Organic', alpha = 0.4)



plt.ylabel('Frequency', fontsize= 25)

plt.xlabel('Average Price', fontsize=20)

plt.legend(loc='upper right')



fig=plt.gcf()

fig.set_size_inches(30.7,15.27)

plt.show()


Organic_sample     = df[(df.year==2017) & (df.type =='organic') & (df['months'] >= 8) & (df['months'] <=10)].AveragePrice.sample(n=50) 

Non_Organic_sample = df[(df.year==2017) & (df.type =='Non-Organic') & (df['months'] >= 8) & (df['months'] <=10)].AveragePrice.sample(n=50)



ttest_ind(Organic_sample, Non_Organic_sample)
Organic2017_sample     = df[(df.year==2017) & (df.type =='organic')].AveragePrice.sample(n=50)

Non_Organic2017_sample  = df[(df.year==2017) & (df.type =='Non-Organic')].AveragePrice.sample(n=50)

ttest_ind(Organic2017_sample,Non_Organic2017_sample )
df2.head()
def monthmap(m):  

    if m=='Jan':  

        return 1

    if m=='Feb':  

        return 2

    if m=='Mar':  

        return 3

    if m=='Apr':  

        return 4

    if m=='May':  

        return 5

    if m=='Jun':  

        return 6

    if m=='Jul':  

        return 7

    if m=='Aug':  

        return 8

    if m=='Sep':  

        return 9

    if m=='Oct':  

        return 10

    if m=='Nov':  

        return 11

    if m=='Dec':  

        return 12


df2['month2'] = df2.month.apply(lambda x: monthmap(x)) 
df2.head()
df2[df2.year==2017].groupby('month2')['4046'].mean().plot() 

df2[df2.year==2017].groupby('month2')['4225'].mean().plot() 

df2[df2.year==2017].groupby('month2')['4770'].mean().plot() 



plt.title('2017: Average volume per month for each PLU code',fontsize=40)

plt.xlabel('Month',fontsize=30)

plt.ylabel('Volume',fontsize=30)

plt.yticks(np.arange(0,2000000, 100000))

plt.legend([4046, 4225, 4770])



fig=plt.gcf()

fig.set_size_inches(30.7,15.27)

plt.show()

s = pd.Series(df['region'])

print(s.nunique())

print(s.unique())