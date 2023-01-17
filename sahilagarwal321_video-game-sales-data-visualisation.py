import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import os
os.listdir("./../input")
data = pd.read_csv("../input/vgsales.csv")
data.shape
data.info()
data.head()
data.columns
data.tail()
df1 = data.groupby(['Year'])
df1_mean = df1['NA_Sales','EU_Sales','JP_Sales'].aggregate(np.mean)
df1_mean.plot()

plt.title('Avg Sales')
plt.hist(data.Year.values,bins=20)

plt.xlabel('Year')

plt.ylabel('frequency')
df3 = data.groupby(['Genre'])
val = df3['NA_Sales'].aggregate(np.mean)
val.plot(kind='bar')

plt.xticks(rotation=30)

plt.xlabel('Genre')

plt.ylabel('NA_Sales')

plt.title('North America Sales as per Genre')
plt.barh(data.Genre,data.EU_Sales)

plt.xlabel('EU_Sales')

plt.ylabel('Genre')

plt.title('European Sales as per Genre')
data.Year.max()
data.describe()
data.head()
sns.barplot(x='Genre',y='Global_Sales',data=data)

plt.title('Global sales as per Genre')

plt.xticks(rotation=45)

plt.show()
sns.jointplot(x='JP_Sales',y='NA_Sales',data=data)
sns.barplot(x='Year',y='Global_Sales',data=data)

plt.title('Global sales per year')

plt.xticks(rotation=45)

plt.show()
sns.heatmap(data.corr())
df_publishers = data.groupby('Publisher')
plot_publishers = df_publishers['NA_Sales','JP_Sales','EU_Sales','Other_Sales'].mean()
plot_publishers.boxplot()
sort_publishers = plot_publishers.sort_values('EU_Sales',ascending=False)

fig = plt.figure(figsize=(12,8))

ax1 = fig.add_subplot(1,4,1)

ax1.set_xticklabels(labels = 'European Union', rotation=90)

sns.barplot(x=plot_publishers.head(5).index, y=sort_publishers.head(5).EU_Sales)

plt.title('European Union')

plt.ylabel('Revenue')

plt.suptitle('Revenues per region',size=22)

sort_publishers = plot_publishers.sort_values('NA_Sales',ascending=False)

ax2 = fig.add_subplot(1,4,2,sharey=ax1)

ax2.set_xticklabels(labels = 'North America', rotation=90)

sns.barplot(x=plot_publishers.head(5).index, y=sort_publishers.head(5).NA_Sales)

plt.title('North America')

plt.ylabel('Revenue')

sort_publishers = plot_publishers.sort_values('JP_Sales',ascending=False)

ax3 = fig.add_subplot(1,4,3,sharey=ax1)

ax3.set_xticklabels(labels = 'Japan', rotation=90)

sns.barplot(x=plot_publishers.head(5).index, y=sort_publishers.head(5).JP_Sales)

plt.title('Japan')

plt.ylabel('Revenue')

sort_publishers = plot_publishers.sort_values('Other_Sales',ascending=False)

ax4 = fig.add_subplot(1,4,4,sharey=ax1)

ax4.set_xticklabels(labels = 'Japan', rotation=90)

sns.barplot(x=plot_publishers.head(5).index, y=sort_publishers.head(5).Other_Sales)

plt.title('Other Regions')

plt.ylabel('Revenue')