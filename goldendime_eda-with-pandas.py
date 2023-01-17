# Import all of the essential libraries for our EDA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['patch.force_edgecolor'] = True
%matplotlib inline
data=pd.read_csv('../input/vgsales.csv',index_col=0)
data.head()
data.info()
data['Year'].value_counts(dropna=False).sort_index()
data.loc[data['Year']==2020, :]
data.loc[5959, 'Year'] = 2010.0
data.loc[5959,]
data['Platform'].value_counts()
data['Genre'].value_counts()
data['Publisher'].nunique()
data['NA_Sales_prop'] = data['NA_Sales'] / data['Global_Sales']
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
fig.set_figheight(12)
fig.set_figwidth(14)
data[data['Year']<2017].groupby('Year')['Global_Sales'].sum().plot(ax=ax1, subplots=True)
ax1.set_title('Total Volume of Sales Globally')
data[data['Year']<2017].groupby('Year')['NA_Sales_prop'].mean().plot(ax=ax2, subplots=True)
ax2.set_title('The Market Share in North America')
data[data['Year']<2017].groupby('Year')['Global_Sales'].mean().plot(ax=ax3, subplots=True)
ax3.set_title('Average Yearly Sales of Each Title')
until_94 = data[(data['Year']<1993)]['Name'].count() / (1993 - 1980)
start_from94 = data[(data['Year']>1993) & (data['Year']<2017)]['Name'].count() / (2017-1994)
start_from94/until_94
grouped = data[(data['Year']>1996) & (data['Year']<2016)].groupby(['Year', 'Genre'])['Global_Sales'].sum().unstack()
grouped.plot(kind='bar', stacked = True, figsize=(14,10))
prop = grouped.divide(grouped.sum(axis=1), axis=0)
prop.plot(kind='area', stacked = True, figsize=(15,10), xticks=[1998, 2002, 2006, 2010, 2014], yticks=np.arange(0, 1.1, step=0.1))
plt.legend(loc='best', bbox_to_anchor=(1.0, 0.5))
#How large is NA sales proportion globally?
data[data['Year']<2017]['NA_Sales_prop'].mean()

grouped = data[data['Year']<2017].groupby(['Platform']).agg({'NA_Sales_prop': 'mean', 'Global_Sales': 'sum'})
grouped[grouped['Global_Sales']>50].sort_values(by = 'NA_Sales_prop').plot(kind='bar', figsize=(15,8), secondary_y=['NA_Sales_prop'])