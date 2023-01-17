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
import seaborn as sb

import matplotlib.pyplot as plt
wine_df =pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv")
wine_df.head()
wine_df.info()
wine_df.tail()
wine_df.iloc[2, 2]
wine_df.drop('Unnamed: 0', axis = 1, inplace = True)
wine_df.shape
wine_df.duplicated().sum()
wine_df.describe()
wine_df['country'].nunique()
wine_df['country'].value_counts()
plt.figure(figsize=[8,8])

sb.countplot(y='country', data= wine_df)
wine_df.head()
#Looking at the distribution of price

binsize= 20

bins= np.arange(0, wine_df['price'].max() +10, binsize)

plt.figure(figsize=[8, 5])

plt.hist(data = wine_df, x = 'price', bins = bins)

plt.title("Wine price")

plt.xlabel('Wine price distribution')

plt.xlim(0, 1000)

plt.show()
#Look at the points distribution

sb.distplot(wine_df['points'], bins = 28);
#Look at the bivariate distributions of price and points

wine_df['variety'].nunique()
sb.regplot(data = wine_df, x= 'points', y ='price', fit_reg= False,

          x_jitter = 0.2, y_jitter = 0.2, scatter_kws = {'alpha' : 1/5});
some_countries=wine_df.loc[(wine_df['country']== 'US')  |(wine_df['country']=='Italy')| (wine_df['country']=='France')]
#Let us compare the price between these countries

#plt.figure(figsize=[10,10])

fig, ax =plt.subplots(ncols=2, figsize=[14,8])

sb.boxplot(data=some_countries, x= 'country', y ='price', ax= ax[0]);

 

sb.boxplot(data=some_countries, x= 'country', y ='points', ax= ax[1]);
some_countries.describe()
sb.regplot(data = some_countries, x= 'points', y ='price', fit_reg= False,

           scatter_kws = {'alpha' : 1/5});
variety =some_countries.loc[some_countries['price'] >=1000]
plt.figure(figsize=[12, 10])

sb.violinplot(x='variety',y = 'price', data = variety);
some_countries.corr()
sb.boxplot(data = variety, x= 'province', y= 'price')
for_heat= variety[['winery','province', 'price']]
for_heat_2= variety[['province', 'price','variety']]
#The wines priced above 1000 dollars come most  from Burgundy province and the winery :Dominane Du Comte

plt.figure(figsize=[10,10])

sb.countplot(data= for_heat, y= 'winery', hue='province');
#for_heat_set.winery =for_heat_set.winery.astype('category')
for_heat.info()
#below_1000.describe()