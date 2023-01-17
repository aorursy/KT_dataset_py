# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from scipy.stats import ttest_ind

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



# Looking at brands, minimum prices, and maximum prices

#data = pd.read_csv("../input/7210_1.csv", low_memory=False)

#data = pd.read_csv("../input/Datafiniti_Womens_Shoes.csv", low_memory=False)

og_data = pd.read_csv("../input/Datafiniti_Womens_Shoes_Jun19.csv", low_memory=False)

og_data = og_data[['brand', 'prices.amountMax']]
# Getting rid of missing data

og_data.dropna(inplace=True)



# Converting brand names to prevent repeat brands

og_data['brand'] = og_data['brand'].str.upper()

og_data['brand'] = og_data['brand'].replace('-', ' ', regex=True)
# Getting the average by first grouping the prices by brand and getting the mean of their highest value

average = og_data.groupby('brand')['prices.amountMax'].mean().sort_values(ascending=False)



average
# Looking into highest prices

og_data.sort_values(by='prices.amountMax', ascending=False)
# It seems that most of the shoes labeled as over $500 as errors. I'll only focus on the shoes listed under $400

under500_data = og_data[og_data['prices.amountMax'] <= 500]
# Getting rid of brands that appear under 5 times

under500min5 = under500_data[under500_data.groupby('brand').brand.transform(len) > 4]

under500min5.brand.value_counts()
# Looking at our new averages

average = under500min5.groupby('brand')['prices.amountMax'].mean().sort_values(ascending=False)



average
# Displaying the top 15 brands with the highest averages

top15 = list(average.head(20)[1:].index)

# Getting the data for those top 15 brands

data15 = under500min5.loc[under500min5['brand'].isin(top15)]

# Let's get that figure!

plt.figure(figsize=(12,20))

sns.set(style='whitegrid', font_scale=2)

sns.barplot(y='brand', x='prices.amountMax', ci=None, data=data15, order=top15).set_title('Top 15 Brands With Highest Price Averages')

plt.xlabel('Price Average (USD)')

plt.ylabel('Brand')
# Now creating a violinplot for those same 15

plt.figure(figsize=(12,20))

sns.violinplot(y='brand', x='prices.amountMax', data=data15, order=top15).set_title('Top 15 Brands With Highest Price Averages')

plt.xlabel('Price Average (USD)')

plt.ylabel('Brand')
# Displaying the top 15 brands with the lowest averages

bottom15 = list(average[-15:].sort_values().index)

# Getting the data for those top 15 brands

lowdata15 = under500_data.loc[under500_data['brand'].isin(bottom15)]

# Let's get plotting!

plt.figure(figsize=(12,20))

sns.set(style='whitegrid', font_scale=2)

sns.barplot(y='brand', x='prices.amountMax', ci=None, data=lowdata15, order=bottom15).set_title('Top 15 Brands With Lowest Price Averages', fontsize=20)

plt.xlim(0,185)

plt.ylabel('Brand', fontsize=20)

plt.xlabel('Price Average (USD)', fontsize=20)
# Now creating a violinplot for the 15 mentioned above

plt.figure(figsize=(12,20))

sns.violinplot(y='brand', x='prices.amountMax', data=lowdata15, order=bottom15).set_title('Top 15 Brands With Lowest Price Averages')

plt.xlabel('Price Average (USD)')

plt.ylabel('Brand')
# Now looking into Asic's price range

asics_data = og_data.loc[og_data.brand == 'ASICS']



plt.figure()

sns.distplot(asics_data['prices.amountMax'], color='m').set_title('Asics Shoe Prices')

plt.xlabel('Price (USD)')

plt.ylabel('Frequency')
