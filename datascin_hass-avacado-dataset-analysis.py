# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
avacado=pd.read_csv('../input/avocado.csv')
print(avacado.columns)
avacado.info()
print(avacado.head())
avacado['type'].unique()
avacado.groupby(['type'])['AveragePrice'].mean()
avacado.groupby(['region','type','year'])['AveragePrice'].mean()
avacado.boxplot(column='AveragePrice',by='type')
regions= avacado['region'].unique()
regions=pd.DataFrame(regions)
print(avacado.describe())
print("Nunique regions:", avacado['region'].nunique())
print("groupby regions :*****")
print(avacado['region'].value_counts())

plt.figure(figsize=(20,8))
plt.plot(avacado.groupby(['region'])['AveragePrice'].mean())
plt.xticks(rotation=90)
plt.show()
avacado.plot.scatter(x='Small Bags', y='AveragePrice',c='DarkBlue')
avacado.plot.scatter(x='Large Bags', y='AveragePrice',c='Blue')
avacado.plot.scatter(x='XLarge Bags', y='AveragePrice',c='Cyan')    
avacado.groupby(['region','type'])['AveragePrice'].mean().unstack().sort_values(by='organic', ascending=False)
# to find the region which sold the most costlier conventional avocados
avacado.groupby(['region','type'])['AveragePrice'].mean().unstack().sort_values(by='conventional', ascending=False)
# to find the highest consuming region of avocados
avacado.groupby('region')['Total Volume'].mean().sort_values()
regions= avacado['region'].unique() 
print(len(regions))
regiondict = {}
for reg in regions:
    regiondict[reg] = pd.DataFrame()
##    following code is same as above
##    regiondict = {reg : pd.DataFrame for reg in regions}

for key in regiondict.keys():
    regiondict[key] = avacado[:][avacado.region == key]

regiondict['Albany'].describe()
regiondict['HartfordSpringfield'].describe()
avacado.plot(kind='scatter', x='year', y='Total Volume', alpha=0.5, color='r')
plt.xlabel('YEAR')
plt.ylabel('VOLUME')
plt.title("YEARLY VOLUME SCATTER")
years= avacado.year.unique()
plt.plot( avacado.groupby('year')['Total Volume'].sum())
plt.title('Yearly Volume')
plt.xlabel('Year', fontsize=14)
plt.xticks(years)
plt.ylabel('Total Volume', fontsize=14)
plt.show()
avacado.groupby('year')['Total Volume'].sum()
avacadoorg= avacado[avacado.type == 'organic']
avacadocon= avacado[avacado.type == 'conventional']
mean_price_conv_dict={}
mean_price_org_dict={}
for yr in years:
    mean_price_org_dict[yr]= avacadoorg [ avacadoorg.year == yr ] .AveragePrice.mean()
    mean_price_conv_dict[yr]= avacadocon [ avacadocon.year == yr ] .AveragePrice.mean()

print(mean_price_org_dict)
print(mean_price_conv_dict)
lists = sorted(mean_price_org_dict.items()) # sorted by key, return a list of tuples
x, y = zip(*lists) # unpack a list of pairs into two tuples
line1= plt.plot(x, y, label= 'organic')

lists2 = sorted(mean_price_conv_dict.items()) # sorted by key, return a list of tuples
x2, y2 = zip(*lists2) # unpack a list of pairs into two tuples
line2= plt.plot(x2, y2, label= 'conventional')

plt.xlabel('Years')
plt.ylabel('Mean Price')
plt.xticks(years)
plt.legend()
plt.title('Comparision between mean price of organic and conventional avocados')
plt.show()
