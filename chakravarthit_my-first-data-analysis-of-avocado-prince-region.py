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

import datetime as dt
df = pd.read_csv ('/kaggle/input/avocado-prices/avocado.csv',index_col = 0)

df.head()
df.shape

df.dtypes

df.head()
df['Date'] = pd.to_datetime (df['Date'])

df.head()
df['region'].value_counts()

df['region'].astype('category',inplace =  True)
df['type'].value_counts()

df['type'].astype('category',inplace = True)
df.head()

df.isna().sum()
df.nunique()
df.head()
print ("\n***** Year 2015 *****\n\n",df [df['year'] == 2015].mean())

print ("\n***** Year 2016 *****\n\n",df [df['year'] == 2016].mean())

print ("\n***** Year 2017 *****\n\n",df [df['year'] == 2017].mean())

print ("\n***** Year 2018 *****\n\n",df [df['year'] == 2018].mean())
df.head()

#del df['Total Price']
df['Total Volume'] = df['Total Volume'].astype('int')

df.insert(loc = 3,column = 'Total Price', value = (df['AveragePrice'] * df ['Total Volume']))
df.info()
df['region'] = df['region'].astype ('category')

df['type'] = df['type'].astype ('category')

df['year'] = pd.to_datetime(df['year'])
df.head()
df.sort_values(['type','region'],ascending = [True,False])
df.head()
#What is the average price of Avocado in 2015 of type Organic in American states

df.head()
mask1 = df['type'] == 'organic'

mask2 = df['year'] == 2015

mask3 = df ['region'] == 'Albany'

df [mask1 & mask2 & mask3].sort_values('Date')

mask = df['type']=='conventional'

g = sns.factorplot('AveragePrice','region',data=df[mask],

                   hue='year',

                   palette='Blues',

                   join=False,

              )





df.groupby('year')['AveragePrice'].max()
df.groupby('year')['AveragePrice'].min()
mask1 = df['type'] == 'organic'

mask2 = df['year'] == 2015

mask3 = df ['region'] == 'Albany'

df_2015= df [mask1 & mask2 & mask3]



%matplotlib inline

df.plot()



df.index = df['Date']
df.head(3)
df.hist(column = 'AveragePrice')
mask1 = df['type'] == 'organic'

mask2 = df['year'] == 2015

mask3 = df ['region'] == 'Albany'

df_2015= df [mask1 & mask2 & mask3]



for val in df['region'].unique():

    print(val)

    mask3 = df ['region'] == val    

    df_2015= df [mask1 & mask2 & mask3]

    print("Minimun Average Price :",df_2015['AveragePrice'].min())

    print("Maximum Average Price :",df_2015['AveragePrice'].max())

    

    
df.head()
#which region has Lowest average prince of avocado in 2015



mask1 = df['type'] == 'organic'

mask2 = df['year'] == 2015

mask3 = df ['region'] == 'Albany'

df_2015 = df [mask1 & mask2 & mask3]

temp2 = 0



for val in df['region'].unique():

    #print(val)

    mask3 = df ['region'] == val

    df_2015 = df [mask1 & mask2 & mask3]

    temp1 = df_2015['AveragePrice'].min()

    if temp1 > temp2:

        temp2 = temp1

        reg = val

        

print ('Min of state : ',reg )

print ('Min Value : ',temp2 )

    

    

    
#which region has highest average prince of avocado in 2015



mask1 = df['type'] == 'organic'

mask2 = df['year'] == 2015

mask3 = df ['region'] == 'Albany'

df_2015 = df [mask1 & mask2 & mask3]

temp2 = 0



for val in df['region'].unique():

    #print(val)

    mask3 = df ['region'] == val

    df_2015 = df [mask1 & mask2 & mask3]

    temp1 = df_2015['AveragePrice'].max()

    if temp1 > temp2:

        temp2 = temp1

        reg = val

        

print ('*** Year 2015 ***')

print ('Max of state : ',reg )

print ('Max Value : ',temp2 )

    

    

    
#which region has highest Total Volume of avocado in 2015



mask1 = df['type'] == 'organic'

mask2 = df['year'] == 2015

mask3 = df ['region'] == 'Albany'

df_2015 = df [mask1 & mask2 & mask3]

temp2 = 0



for val in df['region'].unique():

    #print(val)

    if val == 'TotalUS':

        continue

    mask3 = df ['region'] == val

    df_2015 = df [mask1 & mask2 & mask3]

    temp1 = df_2015['Total Volume'].max()

    if temp1 > temp2:

        temp2 = temp1

        reg = val

        

print ('*** Year 2015 ***')

print ('Max of state : ',reg )

print ('Max Value : ',temp2 )

    

    

    
#which region has Lowest Total Volume of avocado in 2015



mask1 = df['type'] == 'organic'

mask2 = df['year'] == 2015

mask3 = df ['region'] == 'Albany'

df_2015 = df [mask1 & mask2 & mask3]

temp2 = 0



for val in df['region'].unique():

    #print(val)

    if val == 'TotalUS':

        continue

    mask3 = df ['region'] == val

    df_2015 = df [mask1 & mask2 & mask3]

    temp1 = df_2015['Total Volume'].min()

    if temp1 > temp2:

        temp2 = temp1

        reg = val

        

print ('*** Year 2015 ***')

print ('Max of state : ',reg )

print ('Max Value : ',temp2 )

    

    

    
#which region has Lowest Total Volume of avocado in 2015



mask1 = df['type'] == 'organic'

mask2 = df['year'] == 2015

mask3 = df ['region'] == 'Albany'

df_2015 = df [mask1 & mask2 & mask3]

temp2 = 0



for val in df['region'].unique():

    #print(val)

    if val == 'TotalUS':

        continue

    mask3 = df ['region'] == val

    df_2015 = df [mask1 & mask2 & mask3]

    temp1 = df_2015['4225'].max()

    if temp1 > temp2:

        temp2 = temp1

        reg = val

        

print ('*** Year 2015 ***')

print ('Max of state : ',reg )

print ('Max Value of 4225 type: ',temp2 )

    

    

    
#which region has Lowest Total Volume of avocado in 2015



mask1 = df['type'] == 'organic'

mask2 = df['year'] == 2015

mask3 = df ['region'] == 'Albany'

df_2015 = df [mask1 & mask2 & mask3]

temp2 = 0



for val in df['region'].unique():

    #print(val)

    if val == 'TotalUS':

        continue

    mask3 = df ['region'] == val

    df_2015 = df [mask1 & mask2 & mask3]

    temp1 = df_2015['4225'].min()

    if temp1 > temp2:

        temp2 = temp1

        reg = val

        

print ('*** Year 2015 ***')

print ('min of state : ',reg )

print ('min Value of 4225 type: ',temp2 )

    

    

    
df.head()
#type wise min of average price of year 2015 and region.



mask1 = df['type'] == 'conventional'

mask2 = df['year'] == 2015

mask3 = df ['region'] == 'Albany'

df_2015 = df [mask1 & mask2 & mask3]

temp2 = 0



for val in df['region'].unique():

    #print(val)

    if val == 'TotalUS':

        continue

    mask3 = df ['region'] == val

    df_2015 = df [mask1 & mask2 & mask3]

    temp1 = df_2015['AveragePrice'].max()

    if temp1 > temp2:

        temp2 = temp1

        reg = val

        

print ('*** Year 2015 ***')

print ('Max of state : ',reg )

print ('Max Value of Average type convention: ',temp2 )

    

    

    
#type wise min of average price of year 2015 and region.



mask1 = df['type'] == 'conventional'

mask2 = df['year'] == 2015

mask3 = df ['region'] == 'Albany'

df_2015 = df [mask1 & mask2 & mask3]

temp2 = 0



for val in df['region'].unique():

    #print(val)

    if val == 'TotalUS':

        continue

    mask3 = df ['region'] == val

    df_2015 = df [mask1 & mask2 & mask3]

    temp1 = df_2015['AveragePrice'].min()

    if temp1 > temp2:

        temp2 = temp1

        reg = val

        

print ('*** Year 2015 ***')

print ('Min of state : ',reg )

print ('Min Value of Average type convention: ',temp2 )

    

    

    
df.head()
df.head