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
# import data 

fuel_data = pd.read_csv('../input/Crude oil import cost in US dollars per barrel.csv')

#view column names

fuel_data.columns
fuel_data.head()
year_values = fuel_data.iloc[:,0].unique()

year_values
countQ = 0

count_dash = 0

num_row = 0

for n , yr in enumerate(fuel_data.iloc[:,0]): 

    num_row+=1

    if 'Q' in yr:

        countQ+=1

    elif '-' in yr:

        count_dash+=1

        

print('countQ : {0} and count_dash: {1} and num_row: {2}'.format(countQ, count_dash, num_row))

#print('Total abnormal data: {:d}'.format(countQ+count_dash))

print('Fraction of data to fix: {:f}'.format((countQ+count_dash)/num_row))

for n , yr in enumerate(fuel_data.iloc[:,0]): 

    if 'Q' in yr:

        fuel_data.iloc[n,0] = yr[2:6]

fuel_data.iloc[:,0].unique()

from datetime import datetime

for n , yr in enumerate(fuel_data.iloc[:,0]): 

    if '-' in yr:

        test = datetime.strptime(yr, '%b-%y')

        test = pd.to_datetime(test)

        test = test.date()

        fuel_data.iloc[n,0] = test.strftime('%m-%d-%Y')
year_values = fuel_data.iloc[:,0].unique()

year_values
fuel_data.insert(1,'Year', value = None)

fuel_data.head()
for n , yr in enumerate(fuel_data.iloc[:,0]): 

    if '-' in yr:

        fuel_data.iloc[n,1] =  yr[6:]

    else:

         fuel_data.iloc[n,1] = yr

fuel_data.iloc[:,1].unique()
fuel_data.head()
import matplotlib.pyplot as plt

x=fuel_data['Year']

y1=fuel_data['France']

plt.scatter(x,y1)

plt.xticks(rotation='vertical')

plt.legend(['France'])

plt.show()
fuel_data.loc[fuel_data['Year'] == '2018']
avg_prices = fuel_data.groupby('Year').mean()

avg_prices.loc['2018']
avg_prices.head()
years = list(avg_prices.index)

years
avg_prices.insert(0,'Year', value = None)
avg_prices.loc[:,'Year'] = years
avg_prices.head()
import matplotlib.pyplot as plt

x=avg_prices['Year']

y1=avg_prices['France']

y2=avg_prices['Germany']

y3=avg_prices['Italy']

y4=avg_prices['Spain']

y5=avg_prices['UK']

y6=avg_prices['Japan']

y7=avg_prices['Canada']

y8=avg_prices['USA']



plt.plot(x,y1,'-o',x,y2,'-o',x,y3,'-o',\

         x,y4,'-o',x,y5,'-o',x,y6,\

         '-o',x,y7,'-o',x,y8,'-o')

plt.xticks(x, rotation='vertical')

countries = avg_prices.columns [1:]

plt.legend(countries)

plt.show()
avg_prices