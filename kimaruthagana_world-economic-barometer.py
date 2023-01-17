

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import os



snp500 = pd.read_csv('../input/snp500-20132019/snp500.csv',header=0)

news_corp = pd.read_csv('../input/newscorpstockdata/NWS.csv',header=0)

print(snp500.head())

print(news_corp.header())

# Calculating percentage change in dataset

#One can use the formaula of % change but pandas has an inbuilt function.pct_change() which we shall use.

#More about the function here https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.pct_change.html



snp500['pct_change_snp'] = snp500['Adj Close'].pct_change()

news_corp['pct_change_nws'] = news_corp['Adj Close'].pct_change()

relevant_data_snp500 = snp500[ ['Date','pct_change_snp'] ][1:] # since the first value is NAN, we can do away with it

relevant_data_newscorp = news_corp[ ['Date','pct_change_nws'] ]

# correct snp data to start from the same time as newscorp data 2013-06-20

corrected_snp500 = (relevant_data_snp500.loc[relevant_data_snp500['Date'] > '2013-06-19']).reset_index(drop=True)

corrected_nws = (relevant_data_newscorp['pct_change_nws'].loc[relevant_data_newscorp['Date'] > '2013-06-19']).reset_index(drop=True)

plot_data = pd.concat([corrected_snp500,corrected_nws], axis=1)

plot_data['Date'] = pd.to_datetime(plot_data['Date'])

print(plot_data)
plt.figure(figsize=(12,6))



plt.plot(plot_data['Date'],plot_data['pct_change_snp'].cumsum(),color='black',label="S&P500")

plt.plot(plot_data['Date'],plot_data['pct_change_nws'].cumsum(),color='red', label="newscorp")

plt.legend()

plt.grid()