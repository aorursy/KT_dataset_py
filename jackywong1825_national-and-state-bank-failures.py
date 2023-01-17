import numpy as n

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
banks = pd.read_csv('../input/banks.csv')

banks.info()
banks.head()
banks.isnull().sum()
corr = banks[banks.columns].corr()

sns.heatmap(corr,annot = True)

##as you can see on the corr plot Total Deposits and Total Assets have a  strong positive correlation 
banks.get('Institution Type').unique()#these are categorical features which i will transform later


banks.get('Transaction Type').unique()#these are categorical features which i will transform later


banks.get('Charter Type').unique()#'STATE', 'FEDERAL', 'FEDERAL/STATE#these are categorical features which i will transform later


banks.get('Insurance Fund').unique()#these are categorical features which i will transform later
banks.plot.scatter(x = 'Total Assets', y = 'Estimated Loss (2015)')

banks.plot.scatter(x = 'Total Deposits', y = 'Estimated Loss (2015)')

banks.plot.scatter(x = 'Total Assets', y = 'Total Deposits')
sns.stripplot(x = 'Charter Type', y = 'Estimated Loss (2015)', data = banks, jitter = True);
sns.stripplot(x = 'Charter Type', y = 'Total Assets', data = banks, jitter = True);
sns.stripplot(x = 'Charter Type', y = 'Total Deposits', data = banks, jitter = True);
sns.stripplot(x = 'Institution Type', y = 'Estimated Loss (2015)', data = banks, jitter = True);
sns.stripplot(x = 'Insurance Fund', y = 'Estimated Loss (2015)', data = banks, jitter = True);
sns.stripplot(x = 'Insurance Fund', y = 'Total Assets', data = banks, jitter = True);
sns.countplot( y = 'Institution Type', data = banks);
sns.stripplot(x = 'Transaction Type', y = 'Estimated Loss (2015)', data = banks, jitter = True);
sns.stripplot(x = 'Transaction Type', y = 'Total Assets', data = banks, jitter = True);
sns.stripplot(x = 'Transaction Type', y = 'Total Deposits', data = banks, jitter = True);
liquidity = banks['Total Deposits']/banks['Total Assets'] * 100

liquidity = liquidity.dropna()

liquidity.hist(bins = 100)
sns.distplot(liquidity,hist = True, bins = 100)