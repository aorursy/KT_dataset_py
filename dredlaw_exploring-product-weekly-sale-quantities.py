# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Control parameters
autocorr_nlags_max = 13 # Maximum number of lags for calculating autocorrelation
    # Use 13 because 13 weeks is approximately one quarter, i.e. look at repeat purchases within a quarter
corr_thresh = 0.7 # Threshold for correlation score
    # Use >= 0.7 for strongly correlated
param_top_perc = 0.05 # Percent of products with highest average weekly sale quantity to keep for futher analysis
    # Use 0.05 to keep top 40
# Get column names
col_names = pd.read_csv('../input/Sales_Transactions_Dataset_Weekly.csv', nrows=0).columns
#print(col_names)

# Load raw data
# i.e. the product_code and weekly sale amount columns
#raw_data = pd.read_csv('../input/Sales_Transactions_Dataset_Weekly.csv').filter(regex='Product_Code|^W[0-9]+$')
raw_data = pd.read_csv('../input/Sales_Transactions_Dataset_Weekly.csv', 
                       usecols=[col for col in col_names if 'Product_Code' in col or 'W' in col],
                       index_col = 'Product_Code')

# Check shape of data
print(raw_data.shape)
raw_data.head(5)
# Calculate correlation between sale quantities
# Use Spearman method to capture non-linear correlation
product_corr = raw_data.transpose().corr(method='spearman')
print(product_corr.shape)
product_corr.head(5)
# Find and keep high correlations only
high_corr_set = set() # record the individual Product_Code values involved
high_corr_pairs = [] # record the Product_Code pairs and the correlation value

for row in product_corr:
    for col in product_corr.columns:
        if row < col:
            # Get the correlation value, avoiding doing it twice
            corr_val = product_corr.loc[row, col]

            if abs(corr_val) >= corr_thresh:
                # If the correlation value is above the threshold, store it
                #print(row)
                #print(col)
                #print(corr_val)
                high_corr_pairs.append((row, col, corr_val))
                high_corr_set.add(row)
                high_corr_set.add(col)
            
#print(high_corr_list)
# Convert list to a dataframe
high_corr_df = pd.DataFrame(high_corr_pairs, columns=['Product_1', 'Product_2', 'Corr'])
print(f"Number of product pairs with correlation >= {corr_thresh} or <= -{corr_thresh}: {high_corr_df.shape[0]}")
print(f"Number of distinct Product_Code values involved: {len(high_corr_set)}")
high_corr_df.head(5)
# Look at most and least popular products
num_products = raw_data.shape[0]
num_top = math.floor(param_top_perc * num_products)
print(f'Find the most popular and least popular {num_top} products by average weekly sale quantity: ')
print(' ')

# Calculate weekly average sale quantity
avg_weekly_quantity = raw_data.mean(axis=1)

# Find most popular and least popular products
most_pop_products = avg_weekly_quantity.nlargest(n=num_top).index.values.tolist()
least_pop_products = avg_weekly_quantity.nsmallest(n=num_top).index.values.tolist()

# Extract raw data for these products
most_pop_raw_data = raw_data[raw_data.index.isin(most_pop_products)]
least_pop_raw_data = raw_data[raw_data.index.isin(least_pop_products)]

print(f"Have dataframe with {num_top} most popular products: {most_pop_raw_data.shape[0]}")
most_pop_raw_data.head(2)
print(f"Have dataframe with {num_top} least popular products: {least_pop_raw_data.shape[0]}")
least_pop_raw_data.head(2)
# Check whether any most popular or least popular products have a strong correlation with any other product
# Convert the high correlation set to a list
high_corr_list = list(high_corr_set)

# Apply the list - most popular
most_pop_with_strong_corr = most_pop_raw_data[most_pop_raw_data.index.isin(high_corr_list)]
print(f"Number of most popular Product_Code values in the high correlation list: {most_pop_with_strong_corr.shape[0]}")
most_pop_with_strong_corr.head(5)
# Apply the list - least popular
least_pop_with_strong_corr = least_pop_raw_data[least_pop_raw_data.index.isin(high_corr_list)]
print(f"Number of least popular Product_Code values in the high correlation list: {least_pop_with_strong_corr.shape[0]}")
least_pop_with_strong_corr.head(15)

# Least popular products are very rarely bought (multiple weeks of zero items sold) and have high correlation with each other
# As the products only have weekly sale quantity numbers, the analysis of low popularity products ends here
# Examine purchase cycles of most popular products

# Transpose most_pop_raw_data so that the products are listed horizontally (i.e. in the column names)
most_pop_raw_data_horz = most_pop_raw_data.transpose()

# Define function for easy to repeat autocorrelation
def acf(df, nlags_max):
    output_list = []
    for col in df:
        # For each column in the dataframe, extract the column as a series
        series = df[col]
        for nlag in range(1, nlags_max + 1):
            # Calculate autocorrelation for lags from 1 to nlgas_max
            output_list.append((col, nlag, series.autocorr(lag=nlag)))
            
    output_df = pd.DataFrame(output_list, columns=['Product_Code', 'nlag', 'Autocorr'])
    return output_df

# Use the function
most_pop_autocorr = acf(most_pop_raw_data_horz, autocorr_nlags_max)

print(f"Shape of most_pop_autocorr is {most_pop_autocorr.shape}")
most_pop_autocorr.head(autocorr_nlags_max)
# Filter autocorrelation results

# Use 95% confidence interval for identifying statistically significant autocorrelations
acf_ci = 1.96 / math.sqrt(most_pop_raw_data.shape[0])

# Apply filter
sig_most_pop_autocorr = most_pop_autocorr.loc[abs(most_pop_autocorr['Autocorr']) > acf_ci]

print(f"Significance threshold is {acf_ci}")
print(f"Number of siginicant autocorrelations found: {sig_most_pop_autocorr.shape[0]}")
sig_most_pop_autocorr.head(20)