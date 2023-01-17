import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
recent_grads = pd.read_csv('../input/college-earnings-by-major/recent-grads.csv')
recent_grads.iloc[0]
recent_grads.shape
recent_grads.head()
recent_grads.tail()
recent_grads.describe()
raw_data_count = recent_grads.shape[0]
recent_grads = recent_grads.dropna()
cleaned_data_count = recent_grads.shape[0]

print("\nRows before cleaning:",raw_data_count,"\n\n\nRows after cleaning:",cleaned_data_count)
ax1 = recent_grads.plot(x='Sample_size', y='Median', kind='scatter', figsize=(10,5))
ax1.set_title('Sample_size vs. Median')
ax1.set_ylim(10000, 120000)
ax1.set_xlim(-250,5000)
ax2 = recent_grads.plot(x='Sample_size', y='Unemployment_rate', kind='scatter', figsize=(10,5))
ax2.set_title('Sample_size vs. Unemployment_rate')
ax2.set_ylim(-0.025,0.2)
ax2.set_xlim(-250,5000)
ax3 = recent_grads.plot(x='Full_time', y='Median', kind='scatter', figsize=(10,5))
ax3.set_title('Full_time vs. Median')
ax3.set_ylim(10000, 120000)
ax3.set_xlim(-250, 5000)

#All majors make, on average, the same money regardless of the number of full time employees it has projected. 


ax4 = recent_grads.plot(x='ShareWomen', y='Unemployment_rate', kind='scatter', figsize=(10,5))
ax4.set_title('ShareWomen vs. Unemployment_rate')
ax4.set_ylim(-0.025, 0.2)
ax4.set_xlim(-0.1, 1.2)

#Observation: All majors have, on average, the same unemployment rate regardless of the number of women graduated.          


ax5 = recent_grads.plot(x='Men', y='Median', kind='scatter', figsize=(10,5))
ax5.set_title('Men vs. Median')
ax5.set_ylim(10000, 120000)
ax5.set_xlim(-10000, 200000)
ax6 = recent_grads.plot(x='Women', y='Median', kind='scatter', figsize=(10,5))
ax6.set_title('Women vs. Median')
ax6.set_ylim(10000, 120000)
ax6.set_xlim(-10000, 200000)
ax6 = recent_grads.plot(x='Total', y='Median', kind='scatter', figsize=(10,5))
ax6.set_title('Total vs. Median')
ax6.set_ylim(0, 120000)
ax6.set_xlim(0,5000)

#Observation: All majors make, on average, the same money regardless of their popularity.

ax7 = recent_grads.plot(x='ShareWomen', y='Median', kind='scatter', figsize=(10,5))
ax7.set_title('ShareWomen vs. Median')

ax7.set_ylim(0, 120000)
ax7.set_xlim(-0.1, 1.2)

#Observation: Majors with majority of women tend to earn less than male-dominated majors, up to $20k more.

recent_grads['Sample_size'].hist(bins=20, range=(0,4500))
recent_grads['Median'].hist(bins=20, range=(0,120000))

#The most common median salary range is $30k-$35k approx.
recent_grads['Employed'].hist(bins=15, range=(0,max(recent_grads['Employed'])))
recent_grads['Full_time'].hist(bins=20, range=(0,max(recent_grads['Full_time'])))
recent_grads['ShareWomen'].hist(bins=20, range=(0,max(recent_grads['ShareWomen'])))
recent_grads['Unemployment_rate'].hist(bins=20, range=(0,max(recent_grads['Unemployment_rate'])))
recent_grads['Men'].hist(bins=12, range=(0,max(recent_grads['Men'])))
recent_grads['Women'].hist(bins=12, range=(0,max(recent_grads['Women'])))
from pandas.plotting import scatter_matrix
scatter_matrix(recent_grads[['Sample_size','Median']], figsize=(10,10))
scatter_matrix(recent_grads[['Sample_size','Median','Unemployment_rate']], figsize=(10,10))
print(recent_grads[:10].plot.bar(x='Major', y='ShareWomen', legend=False))
print(recent_grads[recent_grads.shape[0]-9:].plot.bar(x='Major', y='ShareWomen', legend=False))
print(recent_grads[:10].plot.bar(x='Major', y='Unemployment_rate', legend=False))
print(recent_grads[recent_grads.shape[0]-9:].plot.bar(x='Major', y='Unemployment_rate', legend=False))
import numpy as np

fig = plt.figure(figsize=(15,25))
ind = np.arange(10)

ax1 = fig.add_subplot(4,1,1)
ax2 = fig.add_subplot(4,1,2)
ax3 = fig.add_subplot(4,1,3)
ax4 = fig.add_subplot(4,1,4)


ax1.bar(ind, recent_grads.loc[:9, 'Men'], 0.35)
ax2.bar(ind, recent_grads.loc[:9, 'Women'], 0.35)
ax3.bar(ind, recent_grads.loc[recent_grads.shape[0]-9:, 'Men'], 0.35)
ax4.bar(ind, recent_grads.loc[recent_grads.shape[0]-9:, 'Women'], 0.35)

ax1.set_xticklabels(recent_grads.loc[:9, 'Major'], rotation=20)
ax2.set_xticklabels(recent_grads.loc[:9, 'Major'], rotation=20)
ax3.set_xticklabels(recent_grads.loc[recent_grads.shape[0]-9:, 'Major'], rotation=20)
ax4.set_xticklabels(recent_grads.loc[recent_grads.shape[0]-9:, 'Major'], rotation=20)

plt.tight_layout()
plt.show()


recent_grads.plot.hexbin(x='Unemployment_rate', y='Median', gridsize=40)
recent_grads.plot.hexbin(x='Women', y='Median', gridsize=40)
recent_grads.plot.hexbin(x='Men', y='Median', gridsize=40)
