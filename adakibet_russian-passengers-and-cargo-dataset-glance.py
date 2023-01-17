# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import sklearn



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('/kaggle/input/russian-passenger-air-service-20072020/russian_air_service_CARGO_AND_PARCELS.csv')

data_1 = pd.read_csv('/kaggle/input/russian-passenger-air-service-20072020/russian_passenger_air_service_2.csv')

print(data.head())

print(data_1.head())
#check for null values in theCARGO and PARCELS data

data.isna().sum()
#check for null values in the passenger data

data_1.isna().sum()
#create a pivot table w/yearly totals for CARGOO/PARCELS data

pivoted_yearly_c = data.pivot_table(index = 'Year', values='Whole year')

print(pivoted_yearly_c.tail())



#create a pivot table w/yearly totals for PASSENGERS data

pivoted_yearly_p = data_1.pivot_table(index = 'Year', values='Whole year')

print(pivoted_yearly_p.tail())



#visualise the yearly totals for both CARGO and PASSENGERS

plt.plot(pivoted_yearly_c, color = 'magenta')

plt.xlabel('Year')

plt.ylabel('Totals')

plt.title('CARGOS AND PARCELS YEARLY - RUSSIAN AIRLINES')

plt.show()



plt.plot(pivoted_yearly_p, color = 'magenta')

plt.xlabel('Year')

plt.ylabel('Totals')

plt.title('PASSENGERS - RUSSIAN AIRLINES')

plt.show()
#checking for correlation between the CARGO and PASSENGERS

pivoted_yearly_c.rename(columns={'Year': 'Year', 'Whole year': 'Total Cargo'}, inplace=True)

pivoted_yearly_p.rename(columns={'Year': 'Year', 'Whole year': 'Total Passengers'}, inplace=True)

print(pivoted_yearly_c.head())

print(pivoted_yearly_p.head())



#merge the tables

new_pivot = pd.merge(pivoted_yearly_p, pivoted_yearly_c, how='inner', on=None, left_on=pivoted_yearly_p.index, right_on=pivoted_yearly_c.index,

left_index=True, right_index=False, sort=True)

new_pivot = new_pivot.drop('key_0', axis = 1)

print(new_pivot.head())



# Compute correlations

corr = new_pivot.corr()

# Exclude duplicate correlations by masking upper right values

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set background color / chart style

sns.set_style(style = 'white')

# Set up  matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))

# Add diverging colormap

cmap = sns.diverging_palette(10, 250, as_cmap=True)

# Draw correlation plot

sns.heatmap(corr, mask=mask, cmap=cmap, 

        square=True,

        linewidths=.5, cbar_kws={"shrink": .5})

#creating a pivot table w/monthly totals

pivot_monthly_c = data.pivot_table(index = 'Year', values = ['January', 'February', 'March', 'April', 'May',

       'June', 'July', 'August', 'September', 'October', 'November', 'December'], aggfunc = sum)

pivot_monthly_c = pivot_monthly_c.drop(2020, axis = 0)  #drop 2020 records.

pivot_monthly_c.loc['Total'] = pivot_monthly_c.sum(axis = 0)    #add the Total index

pivot_monthly_c = pivot_monthly_c.reindex(columns = ['January', 'February', 'March', 'April', 'May',    #reindex the columns

       'June', 'July', 'August', 'September', 'October', 'November', 'December'])





pivot_monthly_p = data_1.pivot_table(index = 'Year', values = ['January', 'February', 'March', 'April', 'May',

       'June', 'July', 'August', 'September', 'October', 'November', 'December'], aggfunc = sum)

pivot_monthly_p = pivot_monthly_p.drop(2020, axis = 0)  #drop 2020 records.

pivot_monthly_p.loc['Total'] = pivot_monthly_p.sum(axis = 0)    #add the Total index

pivot_monthly_p = pivot_monthly_p.reindex(columns = ['January', 'February', 'March', 'April', 'May',   

       'June', 'July', 'August', 'September', 'October', 'November', 'December'])        #reindex the columns

 



print(pivot_monthly_c.tail())

print(pivot_monthly_p.tail())
#acessing the totals row and plotting a bar chart

df = pivot_monthly_c.iloc[[13]] 



plt.figure(figsize=(14,6))

sns.barplot(data = df)

print(df)
df1 = pivot_monthly_p.iloc[[13]]

print(df1)

plt.figure(figsize=(14,6))

sns.barplot(data = df1)