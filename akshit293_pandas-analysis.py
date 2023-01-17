# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#readind data
data=pd.read_csv('../input/avocado-prices/avocado.csv',index_col='Unnamed: 0')
print(data.shape)
data.head(10)
#no missing values
data.describe()
#top 10 regions selling avocados
data.groupby('region').median()['Total Volume'].sort_values(ascending=False)[:10]
#top 10 regions selling avocados with PLU 4046 
data.groupby('region').median()['4046'].sort_values(ascending=False)[:10]
#top 10 regions selling avocados with PLU 4225 
data.groupby('region').median()['4225'].sort_values(ascending=False)[:10]
#top 10 regions selling avocados with PLU 4770 
data.groupby('region').median()['4770'].sort_values(ascending=False)[:10]
#total voulume ,4046,4225,4770 comparison between conventional and organic avocados
data.groupby('type').median()[['Total Volume','4046','4225','4770']]
#each region produces 169 or 166 conventional or organic avocado
print(data.groupby(['region','type']).count()['Date'].unique())

#each region wise
print(data.groupby(['region','type']).count()['Date'])

#only  WestTexNewMexico produces 166 organic , all other regions produces 169 conventional and organic each
print(data.groupby(['region','type']).count()['Date'].value_counts())

#top 10 regions selling avocados
print('top 20 regions selling avocados')
print(data.groupby('region').median()['Total Volume'].sort_values(ascending=False)[:20])

print("****"*10)

#10 most costly avocado selling regions
print('20 most costly avocado selling regions')
print(data.groupby('region').median()['AveragePrice'].sort_values(ascending=False)[:20])


dataanalyse=pd.DataFrame()
dataanalyse['Total Volume']=data.groupby('region').median()['Total Volume']
dataanalyse['Average Price']=data.groupby('region').median()['AveragePrice']
dataanalyse['4770']=data.groupby('region').median()['4770']
dataanalyse['4225']=data.groupby('region').median()['4225']
dataanalyse['4046']=data.groupby('region').median()['4046']

dataanalyse.head(20)
minimum_price=data['AveragePrice'].min()
maximum_price=data['AveragePrice'].max()
print(minimum_price,maximum_price)
data[data['AveragePrice']==minimum_price]
data[data['AveragePrice']==maximum_price]
data[data['region']=='California'].describe()