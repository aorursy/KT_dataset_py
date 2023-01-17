# Data manipulation

import pandas as pd

import numpy as np



# Plotting

import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns



# Finance related operations

from pandas_datareader import data



# Import this to silence a warning when converting data column of a dataframe on the fly

from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()



%matplotlib inline
# Load data

df = pd.read_csv('../input/200-financial-indicators-of-us-stocks-20142018/2014_Financial_Data.csv', index_col=0)



# Drop rows with no information

df.dropna(how='all', inplace=True)
# Get info about dataset

df.info()



# Describe dataset variables

df.describe()
# Plot class distribution

df_class = df['Class'].value_counts()

sns.barplot(np.arange(len(df_class)), df_class)

plt.title('CLASS COUNT', fontsize=20)

plt.show()



# Plot sector distribution

df_sector = df['Sector'].value_counts()

sns.barplot(np.arange(len(df_sector)), df_sector)

plt.xticks(np.arange(len(df_sector)), df_sector.index.values.tolist(), rotation=90)

plt.title('SECTORS COUNT', fontsize=20)

plt.show()
# Extract the columns we need in this step from the dataframe

df_ = df.loc[:, ['Sector', '2015 PRICE VAR [%]']]



# Get list of sectors

sector_list = df_['Sector'].unique()



# Plot the percent price variation for each sector

for sector in sector_list:

    

    temp = df_[df_['Sector'] == sector]



    plt.figure(figsize=(30,5))

    plt.plot(temp['2015 PRICE VAR [%]'])

    plt.title(sector.upper(), fontsize=20)

    plt.show()
# Get stocks that increased more than 500%

gain = 500

top_gainers = df_[df_['2015 PRICE VAR [%]'] >= gain]

top_gainers = top_gainers['2015 PRICE VAR [%]'].sort_values(ascending=False)

print(f'{len(top_gainers)} STOCKS with more than {gain}% gain.')

print()



# Set

date_start = '01-01-2015'

date_end = '12-31-2015'

tickers = top_gainers.index.values.tolist()



for ticker in tickers:

    

    # Pull daily prices for each ticker from Yahoo Finance

    daily_price = data.DataReader(ticker, 'yahoo', date_start, date_end)

    

    # Plot prices with volume

    fig, (ax0, ax1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]})

    

    ax0.plot(daily_price['Adj Close'])

    ax0.set_title(ticker, fontsize=18)

    ax0.set_ylabel('Daily Adj Close $', fontsize=14)

    ax1.plot(daily_price['Volume'])

    ax1.set_ylabel('Volume', fontsize=14)

    ax1.yaxis.set_major_formatter(

            matplotlib.ticker.StrMethodFormatter('{x:.0E}'))



    fig.align_ylabels(ax1)

    fig.tight_layout()

    plt.show()
# Drop those stocks with inorganic gains

inorganic_stocks = tickers[:-2] # all except last 2

df.drop(inorganic_stocks, axis=0, inplace=True)
# Check again for gain-outliers

df_ = df.loc[:, ['Sector', '2015 PRICE VAR [%]']]

sector_list = df_['Sector'].unique()



for sector in sector_list:

    

    temp = df_[df_['Sector'] == sector] # get all data for one sector



    plt.figure(figsize=(30,5))

    plt.plot(temp['2015 PRICE VAR [%]'])

    plt.title(sector.upper(), fontsize=20)

    plt.show()
# Drop columns relative to classification, we will use them later

class_data = df.loc[:, ['Class', '2015 PRICE VAR [%]']]

df.drop(['Class', '2015 PRICE VAR [%]'], inplace=True, axis=1)



# Plot initial status of data quality in terms of nan-values and zero-values

nan_vals = df.isna().sum()

zero_vals = df.isin([0]).sum()

ind = np.arange(df.shape[1])



plt.figure(figsize=(50,10))



plt.subplot(2,1,1)

plt.title('INITIAL INFORMATION ABOUT DATASET', fontsize=22)

plt.bar(ind, nan_vals.values.tolist())

plt.ylabel('NAN-VALUES COUNT', fontsize=18)



plt.subplot(2,1,2)

plt.bar(ind, zero_vals.values.tolist())

plt.ylabel('ZERO-VALUES COUNT', fontsize=18)

plt.xticks(ind, nan_vals.index.values, rotation='90')



plt.show()
# Find count and percent of nan-values, zero-values

total_nans = df.isnull().sum().sort_values(ascending=False)

percent_nans = (df.isnull().sum()/df.isnull().count() * 100).sort_values(ascending=False)

total_zeros = df.isin([0]).sum().sort_values(ascending=False)

percent_zeros = (df.isin([0]).sum()/df.isin([0]).count() * 100).sort_values(ascending=False)

df_nans = pd.concat([total_nans, percent_nans], axis=1, keys=['Total NaN', 'Percent NaN'])

df_zeros = pd.concat([total_zeros, percent_zeros], axis=1, keys=['Total Zeros', 'Percent Zeros'])



# Graphical representation

plt.figure(figsize=(15,5))

plt.bar(np.arange(30), df_nans['Percent NaN'].iloc[:30].values.tolist())

plt.xticks(np.arange(30), df_nans['Percent NaN'].iloc[:30].index.values.tolist(), rotation='90')

plt.ylabel('NAN-Dominance [%]', fontsize=18)

plt.grid(alpha=0.3, axis='y')

plt.show()



plt.figure(figsize=(15,5))

plt.bar(np.arange(30), df_zeros['Percent Zeros'].iloc[:30].values.tolist())

plt.xticks(np.arange(30), df_zeros['Percent Zeros'].iloc[:30].index.values.tolist(), rotation='90')

plt.ylabel('ZEROS-Dominance [%]', fontsize=18)

plt.grid(alpha=0.3, axis='y')

plt.show()
# Find reasonable threshold for nan-values situation

test_nan_level = 0.5

print(df_nans.quantile(test_nan_level))

_, thresh_nan = df_nans.quantile(test_nan_level)



# Find reasonable threshold for zero-values situation

test_zeros_level = 0.6

print(df_zeros.quantile(test_zeros_level))

_, thresh_zeros = df_zeros.quantile(test_zeros_level)
# Clean dataset applying thresholds for both zero values, nan-values

print(f'INITIAL NUMBER OF VARIABLES: {df.shape[1]}')

print()



df_test1 = df.drop((df_nans[df_nans['Percent NaN'] > thresh_nan]).index, 1)

print(f'NUMBER OF VARIABLES AFTER NaN THRESHOLD {thresh_nan:.2f}%: {df_test1.shape[1]}')

print()



df_zeros_postnan = df_zeros.drop((df_nans[df_nans['Percent NaN'] > thresh_nan]).index, axis=0)

df_test2 = df_test1.drop((df_zeros_postnan[df_zeros_postnan['Percent Zeros'] > thresh_zeros]).index, 1)

print(f'NUMBER OF VARIABLES AFTER Zeros THRESHOLD {thresh_zeros:.2f}%: {df_test2.shape[1]}')
# Plot correlation matrix

fig, ax = plt.subplots(figsize=(20,15)) 

sns.heatmap(df_test2.corr(), annot=False, cmap='YlGnBu', vmin=-1, vmax=1, center=0, ax=ax)

plt.show()
# New check on nan values

plt.figure(figsize=(50,10))



plt.subplot(2,1,1)

plt.title('INFORMATION ABOUT DATASET - CLEANED NAN + ZEROS', fontsize=22)

plt.bar(np.arange(df_test2.shape[1]), df_test2.isnull().sum())

plt.ylabel('NAN-VALUES COUNT', fontsize=18)



plt.subplot(2,1,2)

plt.bar(np.arange(df_test2.shape[1]), df_test2.isin([0]).sum())

plt.ylabel('ZERO-VALUES COUNT', fontsize=18)

plt.xticks(np.arange(df_test2.shape[1]), df_test2.columns.values, rotation='90')



plt.show()
# Analyze dataframe

df_test2.describe()
# Cut outliers

top_quantiles = df_test2.quantile(0.97)

outliers_top = (df_test2 > top_quantiles)



low_quantiles = df_test2.quantile(0.03)

outliers_low = (df_test2 < low_quantiles)



df_test2 = df_test2.mask(outliers_top, top_quantiles, axis=1)

df_test2 = df_test2.mask(outliers_low, low_quantiles, axis=1)



# Take a look at the dataframe post-outliers cut

df_test2.describe()
# Replace nan-values with mean value of column, considering each sector individually.

df_test2 = df_test2.groupby(['Sector']).transform(lambda x: x.fillna(x.mean()))
# Plot correlation matrix of output dataset

fig, ax = plt.subplots(figsize=(20,15)) 

sns.heatmap(df_test2.corr(), annot=False, cmap='YlGnBu', vmin=-1, vmax=1, center=0, ax=ax)

plt.show()
# Add the sector column

df_out = df_test2.join(df['Sector'])



# Add back the classification columns

df_out = df_out.join(class_data)



# Print information about dataset

df_out.info()

df_out.describe()