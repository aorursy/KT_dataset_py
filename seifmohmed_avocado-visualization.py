import numpy as np

import pandas as pd

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

df = pd.read_csv('../input/avocado-prices/avocado.csv')
df.head()
df.shape
df.info()

#there is no nan in data
df.drop(columns='year',inplace=True)
# dealing with 'Date' column and splitting it to 'day' , 'Month' and year columns



df[['Year', 'Month','day']]=df.Date.str.split('-', expand = True)
df.head()
df.drop(columns='Date',inplace=True)
df.drop_duplicates(inplace=True)
df.shape
df.region.unique()
df.tail(20)
df.region.value_counts()
sns.barplot(x='type',y='AveragePrice',data=df)
sns.barplot(x='type',y='Total Volume',data=df)
sns.countplot(x='type',data=df)
sns.distplot(df['AveragePrice'])
sns.distplot(df['Total Volume'])
sns.boxplot(x="Year", y="AveragePrice", data=df)
sns.boxplot(x="Month", y="AveragePrice", data=df)
sns.scatterplot(x="Month", y="AveragePrice", data=df)
sns.barplot(x="Month", y="AveragePrice",hue='type', data=df)

#price of orcanic more than conventional
sns.barplot(x="Month", y="Total Volume",hue='type', data=df)

# conventional sold more than orcanic because the brice is little than orcanic
df.head()
df['Unnamed: 0'].value_counts()
df.drop('Unnamed: 0',axis=1,inplace=True)
df.describe()
sns.heatmap(df.corr(),cmap='magma',linecolor='white',linewidths=1)
AVG_REG = df[['region', 'AveragePrice']].groupby('region').agg('mean').sort_values(by = 'AveragePrice', ascending = False).reset_index()
AVG_REG.head()
print('The country with the lowest AveragePrice is ','("',(AVG_REG.min()[0]),'")','with value','("',(AVG_REG.min()[1]),'")')

print('The country with the highest AveragePrice is ','("',(AVG_REG.max()[0]),'")','with value','("',(AVG_REG.max()[1]),'")')
total_REG = df[['region', 'Total Volume']].groupby('region').agg('mean').sort_values(by = 'Total Volume', ascending = False).reset_index()
total_REG.head()
print('The country with the lowest Total Volume is ','("',(total_REG.min()[0]),'")','with value','("',(total_REG.min()[1]),'")')

print('The country with the highest Total Volume is ','("',(total_REG.max()[0]),'")','with value','("',(total_REG.max()[1]),'")')
total_type = df[['type', 'Total Volume']].groupby('type').agg('mean').sort_values(by = 'Total Volume', ascending = False).reset_index()
total_type
ava_type = df[['type', 'AveragePrice']].groupby('type').agg('mean').sort_values(by = 'AveragePrice', ascending = False).reset_index()
ava_type
ava_day = df[['day', 'AveragePrice']].groupby('day').agg('mean').sort_values(by = 'AveragePrice', ascending = False).reset_index()
ava_day.head()

#The most 5 days the AveragePrice is high in them
ava_month = df[['Month', 'AveragePrice']].groupby('Month').agg('mean').sort_values(by = 'AveragePrice', ascending = False).reset_index()
ava_month.head()

#The most 5 months the AveragePrice is high in them
ava_year = df[['Year', 'AveragePrice']].groupby('Year').agg('mean').sort_values(by = 'AveragePrice', ascending = False).reset_index()
ava_year
total_year = df[['Year', 'Total Volume']].groupby('Year').agg('mean').sort_values(by = 'Total Volume', ascending = False).reset_index()
total_year
df.head()
sns.clustermap(df.corr(),cmap='coolwarm',annot=True)
sns.lineplot(x='AveragePrice',y='Total Volume',data=df)
sns.scatterplot(x='AveragePrice',y='Total Volume',data=df)
sns.barplot(x='Month',y='Total Volume',hue='type',data=df,estimator=np.std)
sns.barplot(x='Month',y='Total Bags',hue='type',data=df)
sns.countplot(df.type)
sns.lineplot(x="Month",y='AveragePrice',data=df)

#high price in octoper 
sns.lineplot(x="Month",y='Total Volume',data=df)

# 
#sns.lineplot(x='4046',y='Total Volume',data=df)
#sns.lineplot(x='Total Volume',y='4770',data=df)

#ther is no 
sns.boxplot(x="Year", y="AveragePrice", data=df,palette='rainbow')
sns.boxplot(x="Month", y="AveragePrice", data=df,palette='rainbow')
sns.boxplot(x="day", y="AveragePrice", data=df,palette='rainbow')
plt.figure(figsize=(12,7))

sns.lineplot(x='day',y='AveragePrice',data=df)
#sns.pairplot(df,hue='type',palette='rainbow')
g = sns.PairGrid(df,hue='type')

g.map(plt.scatter)

# orange for orcanic blue for athor
df.head()
#g = sns.JointGrid(x="Total Volume", y="AveragePrice", data=df)

#g = g.plot(sns.regplot, sns.distplot)
sns.lmplot(x='Total Volume',y='4225',data=df,col='type')
#g = sns.JointGrid(x="Total Volume", y="Total Bags", data=df)

#g = g.plot(sns.regplot, sns.distplot)

sns.lmplot(x='Total Volume',y='Total Bags',data=df,col='type')
df.head()
g = sns.FacetGrid(df, col = 'type', row = 'Year', palette = 'RdBu_r',hue = 'region',  height = 3.5, aspect = 2)

g.map(sns.scatterplot, 'Total Volume', 'Total Bags')

g.add_legend()

plt.show()
df.head()
plt.figure(figsize=(10,8))

sns.heatmap(df.corr(),cmap='coolwarm',linecolor='white',annot=True,linewidths=1)