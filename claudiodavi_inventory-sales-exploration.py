import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')
plt.style.use('seaborn-whitegrid')
% matplotlib inline
sales_ratio = pd.read_csv('../input/total-business-inventories-to-sales-ratio_1.csv')
sales_ratio.info()
sales_ratio.drop(['realtime_start', 'realtime_end'], axis=1, inplace=True)
#Indexing on date
sales_ratio['date'] = pd.to_datetime(sales_ratio['date'])
sales_ratio['value'] = pd.to_numeric(sales_ratio['value'], errors='coerce')
sales_ratio.set_index('date', inplace=True)

sales_ratio.head()
sales_ratio.info()
sales_ratio.dropna(inplace=True)
sales_ratio = sales_ratio.asfreq('M', method='ffill')
sales_ratio.info()
# I could have used describe() here, but I find it not intuitive for whomever may be reading your analisys
print(f'Mean {np.round(np.mean(sales_ratio.value),2)}')
print(f'Standard Deviation {np.round(np.std(sales_ratio.value),2)}')
print(f'Median {np.median(sales_ratio.value)}')
print(f'Min {np.min(sales_ratio.value)}')
print(f'Max {np.max(sales_ratio.value)}')
print(f'25% of all values are below: {np.percentile(sales_ratio.value, 25)}')
print(f'50% of all values are below: {np.percentile(sales_ratio.value, 50)}')
print(f'75% of all values are below: {np.percentile(sales_ratio.value, 75)}')
sns.set_style('whitegrid')
sns.set_palette('tab20')
sns.distplot(sales_ratio.value,bins=12, color='b', kde_kws={'color': 'r'})
sns.despine(left=True)
plt.figure(figsize=(10,4))
g = sales_ratio.plot()
sns.despine(left=True)
g.set_title('Inventory/sales ratio through the years')
g.set_xlabel('Year')
g.set_ylabel('Inventory/Sales Ratio')
plt.show()
sales_ratio['month'] = sales_ratio.index.month
sales_ratio.tail(5)
y = sales_ratio['value'].resample('M').mean()
decomposition = sm.tsa.seasonal_decompose(y,model='additive')
decomposition.plot()

g = sns.boxplot(x=sales_ratio['2016-01-01':'2018-01-01'].index.month, y=sales_ratio['2016-01-01':'2018-01-01'].value)
sns.despine(left=True)
g.set_title('Monthly mean of the last 2 years (2016/2017)')
g.set_xlabel('Month')
g.set_ylabel('Mean')
g.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], rotation=45)
plt.show()
plt.figure(figsize=(10,6))
g = sns.barplot(x=sales_ratio['2008-01-01':'2018-01-01'].index.year, y=sales_ratio['2008-01-01':'2018-01-01'].value,ci=None, saturation=0.65, palette='tab20')
g.set_title('Mean of the last 10 years (2008/2017)')
g.set_ylabel('Mean')
g.set_xlabel('Year')
sns.despine(left=True)