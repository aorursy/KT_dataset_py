import pandas as pd

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
data = pd.read_csv('/kaggle/input/avocado-prices/avocado.csv')

data.head()
data.dtypes
data.Date = pd.to_datetime(data.Date)

data.dtypes
data.isnull().sum()
data['AveragePrice'].agg(['min','max','mean','std'])
data['month_name'] = data.Date.dt.month_name()

data.head()
date_price = data[['year','month_name','AveragePrice']]

# Checking which years we have in our data

date_price.year.unique()
# Convertig 'month_names' in category data type ir order to plot de graph sorted by month

month_ordered = ['January','February','March','April','May','June','July','August','September','October','November','December']

date_price['month_name'] = pd.Categorical(date_price['month_name'], categories=month_ordered, ordered=True)



# Slicing by year

price2015 = date_price.loc[date_price['year'] == 2015].groupby('month_name').mean()

price2016 = date_price.loc[date_price['year'] == 2016].groupby('month_name').mean()

price2017 = date_price.loc[date_price['year'] == 2017].groupby('month_name').mean()

price2018 = date_price.loc[date_price['year'] == 2018].groupby('month_name').mean()
sns.set_style('darkgrid')

plt.figure(figsize=(12,6))

sns.lineplot(x= price2015.index, y= price2015.AveragePrice, label='2015')

sns.lineplot(x= price2016.index, y= price2016.AveragePrice, label='2016')

sns.lineplot(x= price2017.index, y= price2017.AveragePrice, label='2017')

sns.lineplot(x= price2018.index, y= price2018.AveragePrice, label='2018')

plt.xlabel('Month')
sns.set_style('darkgrid')

fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(20,6))

fig.suptitle('Average Price distribution per Type of Avocado')



# Swarm plot Axes

ax[0].set_title('Average Prices per Type')

sns.swarmplot(ax=ax[0], x= data.type, y= data.AveragePrice)

# KDE plot Axes

ax[1].set_title('KDE for Average Price')

sns.kdeplot(ax=ax[1], data=data.loc[data['type'] == 'conventional']['AveragePrice'], shade=True, label='Conventional')

sns.kdeplot(ax=ax[1], data=data.loc[data['type'] == 'organic']['AveragePrice'], shade=True, label='Organic')
region = data[['type','region','Total Volume']]

region = region.loc[~region['region'].isin(['TotalUS','West','SouthCentral','Northeast','Southeast','Plains','GreatLakes','Midsouth region',\

                                            'Midsouth'])]


region_total = region[['region','Total Volume']].groupby('region').sum().sort_values(by='Total Volume', ascending=False).iloc[0:5]

region_organic = region.loc[(region.type == 'organic') & (region.region.isin(list(region_total.index)))][['region','Total Volume']]

region_organic = region_organic.groupby('region').sum().sort_values(by='Total Volume', ascending=False)

plt.figure(figsize=(14,6))

sns.barplot(x=region_total.index, y=region_total['Total Volume'], color='purple', label='Conventional')

sns.barplot(x=region_organic.index, y=region_organic['Total Volume'], color='blue', label='Organic')

plt.legend(fontsize=14)
california = data.loc[data.region == 'California']

california['month_name'] = pd.Categorical(california['month_name'], categories=month_ordered, ordered=True)

fig, ax= plt.subplots(1,2, figsize=(25,6))

plt.suptitle('Price Variation x Amount of Avocado sold in California')

sns.lineplot(ax=ax[0], x=california['month_name'], y = california['AveragePrice'])

sns.lineplot(ax=ax[1], x=california['month_name'], y = california['Total Volume'], color='purple')