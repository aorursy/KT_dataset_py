# Import Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# This is to supress the warning messages (if any) generated in our code
import warnings
warnings.filterwarnings('ignore')

# Comment this if the data visualisations doesn't work on your side
%matplotlib inline

# We are using whitegrid style for our seaborn plots. This is like the most basic one
sns.set_style(style = 'whitegrid')
dataset = pd.read_csv('../input/train.csv')
nrow, ncol = dataset.shape
nrow, ncol
#Let's look at first few rows of our dataset
dataset.head(3)
dataset.info()
ds_cat = dataset.select_dtypes(include = 'object').copy()
ds_cat.head(2)
ds_cat['MSZoning'].unique()
len(ds_cat['MSZoning'].unique())
ds_cat['MSZoning'].nunique()
ds_cat['MSZoning'].isnull().sum()
ds_cat['MSZoning'].isnull().sum()/ nrow

#Let's multiple by 100 and keep only 1 decimal places
(ds_cat['MSZoning'].isnull().sum()/ nrow).round(3)*100
sns.countplot(data = ds_cat, x = 'MSZoning')
ds_cat['SalePrice'] = dataset.loc[ds_cat.index, 'SalePrice'].copy()
len(ds_cat.columns) # 43 means 11 rows will be needed
sns.boxplot(data = ds_cat, x='MSZoning', y='SalePrice')
sns.violinplot(data = ds_cat, x='MSZoning', y='SalePrice')
sns.swarmplot(data = ds_cat, x='MSZoning', y='SalePrice')
sns.violinplot(data = ds_cat, x='MSZoning', y='SalePrice')
sns.swarmplot(data = ds_cat, x='MSZoning', y='SalePrice', color = 'k', alpha = 0.6)
fig = plt.figure()

ax1 = fig.add_subplot(2,1,1) 
sns.countplot(data = ds_cat, x = 'MSZoning', ax = ax1)

ax2 = fig.add_subplot(2,1,2) 
sns.boxplot(data = ds_cat, x='MSZoning', y='SalePrice' , ax = ax2)
#sns.violinplot(data = ds_cat, x='MSZoning', y='SalePrice' , ax = ax2)

# Try using VIOLIN PLOT as well. This can give you a lot of details on your underlying data
fig = plt.figure(figsize = (15,10))

ax1 = fig.add_subplot(2,3,1)
sns.countplot(data = ds_cat, x = 'MSZoning', ax=ax1)

ax2 = fig.add_subplot(2,3,2)
sns.countplot(data = ds_cat, x = 'LotShape', ax=ax2)

ax3 = fig.add_subplot(2,3,3)
sns.countplot(data = ds_cat, x = 'LotConfig', ax=ax3)

ax4 = fig.add_subplot(2,3,4)
sns.boxplot(data = ds_cat, x = 'MSZoning', y = 'SalePrice' , ax=ax4)
#sns.violinplot(data = ds_cat, x = 'MSZoning', y = 'SalePrice' , ax=ax4)
#sns.swarmplot(data = ds_cat, x = 'MSZoning', y='SalePrice', color = 'k', alpha = 0.4, ax=ax4  )

ax5 = fig.add_subplot(2,3,5)
sns.boxplot(data = ds_cat, x = 'LotShape', y = 'SalePrice', ax=ax5)
#sns.violinplot(data = ds_cat, x = 'LotShape', y = 'SalePrice', ax=ax5)
#sns.swarmplot(data = ds_cat, x = 'LotShape', y='SalePrice', color = 'k', alpha = 0.4, ax=ax5  )

ax6 = fig.add_subplot(2,3,6)
sns.boxplot(data = ds_cat, x = 'LotConfig', y = 'SalePrice', ax=ax6)
#sns.violinplot(data = ds_cat, x = 'LotConfig', y = 'SalePrice', ax=ax6)
#sns.swarmplot(data = ds_cat, x = 'LotConfig', y='SalePrice', color = 'k', alpha = 0.4, ax=ax6  )