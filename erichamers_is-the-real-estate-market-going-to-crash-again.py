# Math and data modules

import numpy as np 

import pandas as pd



# Data visualization modules

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set_style('darkgrid') # setting the style of the plots



pd.options.display.max_columns = 999 

pd.options.display.max_rows = 999 



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
data = pd.read_csv('../input/State_time_series.csv')
data.head()
data['Date'] = pd.to_datetime(data['Date'])

data['Year'] = data['Date'].apply(lambda x: x.year)
data.groupby(data['Year'])['MedianSoldPrice_AllHomes'].mean().dropna().plot(linewidth=4, figsize=(15, 6))

plt.title('Median Sold Prices by Year', fontsize=14)

plt.ylabel('Price\n')

plt.show()
states = data.groupby(data['RegionName'])['MedianSoldPrice_AllHomes'].mean().dropna().sort_values(ascending=False).index

values = data.groupby(data['RegionName'])['MedianSoldPrice_AllHomes'].mean().dropna().sort_values(ascending=False).values

plt.figure(figsize=(15, 15))

sns.barplot(y=states, x=values, color='blue')

plt.title('Median Price by State', fontsize=14)

plt.xlabel('Price', fontsize=12)

plt.ylabel('State', fontsize=12)

print('Highest Median Price ${:,.2f} in {}'.format(round(values[0], 2), states[0]))

print('Lowest Median Price ${:,.2f} in {}'.format(round(values[-1], 2), states[-1]))
data.groupby(data['Year'])['MedianRentalPrice_AllHomes'].mean().dropna().plot(linewidth=4, figsize=(15, 6))

plt.title('Median Rental Prices', fontsize=14)

plt.ylabel('Price\n')

plt.show()
states = data.groupby(data['RegionName'])['MedianRentalPrice_AllHomes'].mean().dropna().sort_values(ascending=False).index

values = data.groupby(data['RegionName'])['MedianRentalPrice_AllHomes'].mean().dropna().sort_values(ascending=False).values

plt.figure(figsize=(15, 15))

sns.barplot(y=states, x=values, color='blue')

plt.title('Rental Price by State', fontsize=14)

plt.xlabel('Price', fontsize=12)

plt.ylabel('State', fontsize=12)

print('Highest Median Price ${:,.2f} in {}'.format(round(values[0], 2), states[0]))

print('Lowest Median Price ${} in {}'.format(round(values[-1], 2), states[-1]))
plt.title('Days to Sell a House', fontsize=14)

data.groupby(data['Year'])['DaysOnZillow_AllHomes'].mean().dropna().plot(linewidth=4, figsize=(15, 6))

plt.ylabel('Days\n')

plt.show()
states = data.groupby(data['RegionName'])['DaysOnZillow_AllHomes'].mean().dropna().sort_values(ascending=False).index

values = data.groupby(data['RegionName'])['DaysOnZillow_AllHomes'].mean().dropna().sort_values(ascending=False).values

plt.figure(figsize=(13, 15))

sns.barplot(y=states, x=values, color='blue')

plt.title('Median Days on Zillow', fontsize=14)

plt.xlabel('Days', fontsize=12)

plt.ylabel('State', fontsize=12)

print('Hardest to sell in {} with {} days'.format(states[0], round(values[0], 2)))

print('Easiest to sell in {} with {} days'.format(states[-1], round(values[-1], 2)))
f, ax = plt.subplots(2, 1, figsize=(15, 12))

data.groupby(data['Year'])['HomesSoldAsForeclosuresRatio_AllHomes'].mean().plot(linewidth=4, ax=ax[0])

ax[0].set_title('Homes Sold as Foreclosure Ratio')

ax[0].set_ylabel('Ratio')

data.groupby(data['Year'])['MedianSoldPrice_AllHomes'].mean().plot(linewidth=4, ax=ax[1])

ax[1].set_title('Median Sold Price')

ax[1].set_ylabel('Price')

plt.show()
states = data.groupby(data['RegionName'])['HomesSoldAsForeclosuresRatio_AllHomes'].mean().dropna().sort_values(ascending=False).index

values = data.groupby(data['RegionName'])['HomesSoldAsForeclosuresRatio_AllHomes'].mean().dropna().sort_values(ascending=False).values

plt.figure(figsize=(15, 15))

sns.barplot(y=states, x=values, color='blue')

plt.title('Homes Sold as Foreclosure Ratio', fontsize=14)

plt.xlabel('Ratio', fontsize=12)

plt.ylabel('State', fontsize=12)

print('Highest foreclosure ratio of {} in {}'.format(round(values[0], 2), states[0]))

print('Lowest foreclosure ratio of {} in {}'.format(round(values[-1], 2), states[-1]))
f, ax = plt.subplots(2, 2, figsize=(15, 12))



data.groupby(data['Year'])['PctOfHomesDecreasingInValues_AllHomes'].mean().dropna().plot(linewidth=4, ax=ax[0, 0])

data.groupby(data['Year'])['PctOfHomesIncreasingInValues_AllHomes'].mean().dropna().plot(linewidth=4, ax=ax[1, 0])

data.groupby(data['Year'])['PctOfHomesSellingForGain_AllHomes'].mean().dropna().plot(linewidth=4, ax=ax[1, 1])

data.groupby(data['Year'])['PctOfHomesSellingForLoss_AllHomes'].mean().dropna().plot(linewidth=4, ax=ax[0, 1])

ax[0, 0].set_title('Percentage Of Homes Decreasing in Value')

ax[0, 1].set_title('Percentage Of Homes Selling for Loss')

ax[1, 0].set_title('Percentage Of Homes Increasing in Value')

ax[1, 1].set_title('Percentage Of Homes Selling for Gain')

plt.show()
max_price = data.groupby(data['Year'])['MedianSoldPrice_AllHomes'].mean().max()

max_year  = data.groupby(data['Year'])['MedianSoldPrice_AllHomes'].mean().argmax()

min_price = data.groupby(data['Year'])['MedianSoldPrice_AllHomes'].mean().min()

min_year = data.groupby(data['Year'])['MedianSoldPrice_AllHomes'].mean().argmin()

print('Max median price of ${:,.2f} in {}'.format(max_price, max_year))

print('Min median price of ${:,.2f} in {}'.format(min_price, min_year))
min_year_after07  = data.groupby(data['Year'])['MedianSoldPrice_AllHomes'].mean()[11:].argmin()

min_price_after07 = data.groupby(data['Year'])['MedianSoldPrice_AllHomes'].mean()[11:].min()

print('Min median price after \'07 of ${:,.2f} in {}'.format(min_price_after07, min_year_after07))
values_07 = data[data['Year'] == 2007].groupby(data['RegionName'])['MedianSoldPrice_AllHomes'].mean().dropna().values

estates   = data[data['Year'] == 2007].groupby(data['RegionName'])['MedianSoldPrice_AllHomes'].mean().dropna().index

values_12 = data[data['Year'] == 2012].groupby(data['RegionName'])['MedianSoldPrice_AllHomes'].mean().dropna().values

df = pd.DataFrame({'2007 Price': values_07, 

                   '2012 Price': values_12}, index=estates)

df['Variation'] = round((df['2012 Price'] - df['2007 Price'])/df['2007 Price'], 2)
estates = df['Variation'].sort_values(ascending=True).head(5).index

values  = df['Variation'].sort_values(ascending=True).head(5).values

print(df['Variation'].sort_values(ascending=True)[:5])

plt.figure(figsize=(10, 8))

plt.title('Price Variation 07 x 12 by State',fontsize=14)

plt.xticks(fontsize=8)

plt.yticks(fontsize=8)

plt.ylabel('Price Variation\n', fontsize=12)

sns.barplot(x=estates, y=values)

plt.xlabel('\nRegion Name', fontsize=12)

plt.ylim(0, -0.6)

plt.show()
prices = [data[data['Year'] == 2006]['MedianSoldPrice_AllHomes'].mean(),

         data[data['Year'] == 2008]['MedianSoldPrice_AllHomes'].mean(),

         data[data['Year'] == 2012]['MedianSoldPrice_AllHomes'].mean(),

         data[data['Year'] == 2016]['MedianSoldPrice_AllHomes'].mean()]

years = [2006, 2008, 2012, 2016]

print('Median prices in 2006: ${:,.2f}'.format(round(prices[0], 2)))

print('Median prices in 2008: ${:,.2f}'.format(round(prices[1], 2)))

print('Median prices at 2012: ${:,.2f}'.format(round(prices[2], 2)))

print('Median prices in 2016: ${:,.2f}'.format(round(prices[3], 2)))

plt.figure(figsize=(10, 8))

sns.barplot(y=prices, x=years)

plt.title('Price by Year', fontsize=14)

plt.xticks(fontsize=12)

plt.xlabel('Year', fontsize=12)

plt.yticks(fontsize=12)

plt.ylabel('Price\n', fontsize=12)

plt.show()