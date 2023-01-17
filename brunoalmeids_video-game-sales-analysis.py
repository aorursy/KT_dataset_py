import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from  matplotlib.ticker import FuncFormatter
%matplotlib inline
vgsales = pd.read_csv("../input/vgsales.csv")
vgsales.head()
dic = [['NA Sales' , 4392.950000000001], ['EU Sales', 2434.1299999999997], ['JP Sales' , 1291.0200000000002],
       ['Other Sales' , 797.7500000000001], ['Global Sales', 8920.44]]
vgsales_from_dic = pd.DataFrame(data=dic,index = [0,1,2,3,4],columns = ['Region','Total_Sales'])
fig = plt.figure()
axes = sns.barplot(x='Region',y='Total_Sales',data=vgsales_from_dic,palette="coolwarm",order = vgsales_from_dic
                   .set_index('Region')['Total_Sales'].sort_values(ascending=True).reset_index()['Region'].unique())
axes.set_ylabel('Total Sales')
# Global Sales per Console
fig = plt.figure(figsize=(14,8))
ax = sns.barplot(x='Platform',y='Global_Sales',data=vgsales,estimator=np.sum,order = vgsales.groupby('Platform')
                 ['Global_Sales'].sum().sort_values(ascending=False).reset_index()['Platform'].unique(),
                 palette='nipy_spectral',alpha=0.8)
ax.set_ylabel('Global Sales')
# Games per Console
fig = plt.figure(figsize=(14,8))
ax = sns.countplot(x='Platform',data=vgsales,order = vgsales.groupby('Platform')
                 ['Global_Sales'].count().sort_values(ascending=False).reset_index()['Platform'].unique(),
                 palette='nipy_spectral',alpha=0.8)
ax.set_ylabel('Video Games Count')
# 20 Best-selling Games of History
vgsales[['Name','Global_Sales']].set_index('Name')

fig = plt.figure(figsize=(12,9))
ax = sns.barplot(x='Name',y='Global_Sales',data=vgsales.head(20),order = vgsales['Name'].head(20).unique(),
                 palette='nipy_spectral',alpha=0.8)
ax.set_ylabel('Global Sales per Game')
plt.xticks(rotation=80)
plt.tight_layout()
# Games release density on time
plt.figure(figsize=(14,6))
sns.distplot(vgsales.sort_values('Year')['Year'].dropna(),kde=True,bins=40,color='green')
# Titles per Publisher
vgpublisher = vgsales.groupby(vgsales['Publisher']).count()['Global_Sales'].sort_values(ascending = False).reset_index().head(20)
    
fig = plt.figure(figsize=(12,9))
ax = sns.barplot(x='Publisher',y='Global_Sales',data=vgpublisher,order = vgpublisher['Publisher'].unique(),
                 palette='nipy_spectral',alpha=0.8)
ax.set_ylabel('Video Games Count')
plt.xticks(rotation=80)
plt.tight_layout()
# Sales per Publisher
fig = plt.figure(figsize=(12,9))
ax = sns.barplot(x='Publisher',y='Global_Sales',data=vgsales.groupby(vgsales['Publisher'])['Global_Sales'].sum()
                 .sort_values(ascending = False).head(20).reset_index(),order = vgsales.groupby(vgsales['Publisher'])
                 ['Global_Sales'].sum().sort_values(ascending = False).head(20).reset_index()['Publisher'].unique(),
                 palette='nipy_spectral',alpha=0.8)
ax.set_ylabel('Global Sales')
plt.xticks(rotation=80)
plt.tight_layout()
# Sales per Year by Region - graph
fig = plt.figure(figsize=(12,6))
d = {'Global_Sales':'olivedrab','NA_Sales':'slateblue','EU_Sales':'lightcoral','JP_Sales':'peru','Other_Sales':'slategrey'}

for i in ['Global_Sales','NA_Sales','EU_Sales','JP_Sales','Other_Sales']:  # Continue...
    axes = sns.tsplot(data=vgsales.dropna().groupby('Year').sum().reset_index()[i],condition=i.split('_')[0],color=d[i])
plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, _: int(x+1980)))
plt.tight_layout()
# Correlation Map
plt.figure(figsize=(10,8))
sns.heatmap(vgsales.corr(),cmap='coolwarm',annot=True,linecolor='black',linewidths=3)
# Sales per Gender
plt.figure(figsize=(20,8))

axes = sns.heatmap(vgsales.pivot_table(index='Genre',columns='Year',values='Global_Sales',aggfunc=np.sum).fillna(0)
                   ,cmap='coolwarm',annot=False,linecolor='black',linewidths=3,robust=True)
plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, _: int(x+1980)))
# Publishes per Gender
plt.figure(figsize=(20,8))

axes = sns.heatmap(vgsales.pivot_table(index='Genre',columns='Year',values='Global_Sales',aggfunc=np.count_nonzero).fillna(0)
                   ,cmap='coolwarm',annot=False,linecolor='black',linewidths=3,robust=True)
plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, _: int(x+1980)))
# Sales per Gender - EU
plt.figure(figsize=(20,8))

axes = sns.heatmap(vgsales.pivot_table(index='Genre',columns='Year',values='EU_Sales',aggfunc=np.sum).fillna(0)
                   ,cmap='coolwarm',annot=False,linecolor='black',linewidths=3,robust=True)
plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, _: int(x+1980)))
# Sales per Gender - JP
plt.figure(figsize=(20,8))

axes = sns.heatmap(vgsales.pivot_table(index='Genre',columns='Year',values='JP_Sales',aggfunc=np.sum).fillna(0)
                   ,cmap='coolwarm',annot=False,linecolor='black',linewidths=3,robust=True)
plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, _: int(x+1980)))
# Sales per Gender - NA
plt.figure(figsize=(20,8))

axes = sns.heatmap(vgsales.pivot_table(index='Genre',columns='Year',values='NA_Sales',aggfunc=np.sum).fillna(0)
                   ,cmap='coolwarm',annot=False,linecolor='black',linewidths=3,robust=True)
plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, _: int(x+1980)))
# Sales per Gender - Other
plt.figure(figsize=(20,8))

axes = sns.heatmap(vgsales.pivot_table(index='Genre',columns='Year',values='Other_Sales',aggfunc=np.sum).fillna(0)
                   ,cmap='coolwarm',annot=False,linecolor='black',linewidths=3,robust=True)
plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, _: int(x+1980)))
# Considering 2015 as the last year with solid data to compare
# Removing the top 20 from 2012 to view smaller titles and publishers

plt.figure(figsize=(8,8))
axes = sns.heatmap(vgsales[vgsales['Year'] == 2015][20:].groupby('Genre').sum().drop(['Rank','Year'],axis=1),
                   cmap='coolwarm',annot=True,linecolor='black',linewidths=3,robust=True)
