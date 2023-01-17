import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import os

from sklearn import preprocessing



# 1.3 Class for applying multiple data transformation jobs

from sklearn.compose import ColumnTransformer as ct

# 1.4 Scale numeric data

from sklearn.preprocessing import StandardScaler as ss

# 1.5 One hot encode data--Convert to dummy

from sklearn.preprocessing import OneHotEncoder as ohe

# 1.6 For clustering

from sklearn.cluster import KMeans  

# for 3D plot

from mpl_toolkits import mplot3d
pd.options.display.max_columns = 200

sales = pd.read_csv("../input/vgsales.csv")
# Get to know about the data etc.

sales.shape
sales.columns
sales.dtypes

sales.dtypes.value_counts() 
#Look at data

sales.head() 
sales.tail()
# Missing data

sales.isnull().values.any()
sales.isnull().sum()
sales.info()
table_sales = pd.pivot_table(sales,values=['Global_Sales'],index=['Year'],columns=['Genre'],aggfunc='max',margins=True)



plt.figure(figsize=(21,17))

sns.heatmap(table_sales['Global_Sales'],linewidths=.5,annot=True,vmin=10,vmax=60, cmap='PuBu')

plt.title('Max Global_Sales of games')



#Depending on the values of vmin and vmax the color of values are varing.

#Margins=False is default, if we change to 'True' then 'ALL' row and column is displaying.

#Genre whisch has maximum sales is 'Sports' in the year 2006.

#If target sales is 60million, based on min and max values we can easily interpret the below graph whith higlightes.  

#Min EU sales



table_sales = pd.pivot_table(sales,values=['Global_Sales', 'EU_Sales'],index=['Year'],columns=['Genre'],aggfunc='sum',margins=False)



plt.figure(figsize=(20,17))

sns.heatmap(table_sales['EU_Sales'],linewidths=.5,annot=True,vmin=0.01,cmap='PuBu')

plt.title('Min EU_Sales of games')
def top(df, n = 1, column = 'Year'):

    return df.sort_values(by=column)[-n:]

sales.groupby(['Year'], group_keys=False).apply(top)[['Year', 'Name', 'Global_Sales', 'Genre', 'Platform', 'Publisher']]
#Categorical distribution plots:

sns.catplot(x='Genre',y='NA_Sales', data=sales);

plt.xticks(rotation=50);
data_new1=sales[sales['Year'] > 2016.0 ]

sns.catplot(x="Genre", y="Other_Sales", hue="Year", kind="box", data=data_new1)

plt.xticks(rotation=50);
data_new2=sales[sales['Year'] < 2000.0 ]

sns.catplot(x="NA_Sales", y="Genre",  hue='Year', kind="violin", bw=.05, cut=0, data=data_new2)
#pairgrid

g = sns.PairGrid(sales,vars=["NA_Sales", "Year"], hue="Genre")

g.map_diag(plt.hist)

g.map_offdiag(plt.scatter)

g.add_legend();
#pointplot takes a dataframe : Highest Publisher Revenue;



table = sales.pivot_table('Global_Sales', index='Publisher', columns='Year', aggfunc='sum')

publishers = table.idxmax()

sales = table.max()

years = table.columns.astype(int)

data = pd.concat([publishers, sales], axis=1)

data.columns = ['Publisher', 'Global Sales']

plt.figure(figsize=(12,8))

ax = sns.pointplot(y = 'Global Sales', x = years , hue='Publisher', data=data, size=15)

ax.set_xlabel(xlabel='Year', fontsize=16)

ax.set_ylabel(ylabel='Global Sales Per Year', fontsize=16)

ax.set_title(label='Highest Publisher Revenue in $ Millions Per Year', fontsize=20)

ax.set_xticklabels(labels = years, fontsize=12, rotation=50)

plt.show();
#top 10 publiser:barplot

sales = pd.read_csv("../input/vgsales.csv")



data = sales.groupby(['Publisher']).count().iloc[:,0]

data = pd.DataFrame(data.sort_values(ascending=False))[0:10]

publishers = data.index

data.columns = ['Releases']

colors = sns.color_palette("spring", len(data))

plt.figure(figsize=(16,8))

ax = sns.barplot(y = publishers , x = 'Releases', data=data, orient='V', palette=colors)

ax.set_xlabel(xlabel='Number of Releases', fontsize=8)

ax.set_ylabel(ylabel='Publisher', fontsize=8)

ax.set_title(label='Top 10 Total Publisher Games Released', fontsize=10)

ax.set_yticklabels(labels = publishers, fontsize=8)

plt.show();

#Number of Years:

sns.countplot(x = 'Year', data=sales) 

plt.xticks(rotation=90)
#barplot showing average of NA_Sales for each Genre by Year, filtered data year > 2000.0:

data_new=sales[sales['Year'] > 2015.0]

data_new.info()

sns.barplot(x='Genre',y='NA_Sales',  hue='Year',  data=data_new)

#plt.legend(loc='Upper left')

plt.xticks(rotation=70)
#barplot showing average of Global_Sales for each year



sns.barplot(x='Year', y='Global_Sales',  data=sales) 

plt.xticks(rotation=90)
#regplot:

fig, axs = plt.subplots(ncols=3,figsize=(20,6))

sns.regplot(x='Year', y='NA_Sales', data=sales , ax=axs[0])

axs[0].set_title('Puzzle', fontsize=18)



sns.regplot(x='Year', y='NA_Sales', data=sales, ax=axs[1])

axs[1].set_title('Platform', fontsize=18)



sns.regplot(x='Year',y='NA_Sales', data=sales, ax=axs[2])

axs[2].set_title('Mic', fontsize=18)

#plot sales for each Year with Genre using boxplot 

sns.catplot(x="Genre", y="Year", kind="box", data=sales)

plt.xticks(rotation=50);



data_new1=sales[sales['Genre'] == ('Misc' 'Shooter')]

data_new2=data_new1[data_new1['Platform']=='GB']

data_new2.info()
#Scatter plot showing the Other_Sales for each Genre

sns.scatterplot(x="Year", y="Other_Sales", data=sales)
#histogram showing the Global_Sales using kde and rug

sns.distplot(sales['JP_Sales'],  kde=True,rug=True,bins=5)
#pairplot for year > 2000 dataframe

sns.pairplot(data_new)
#3D plot to show 

fig = plt.figure()

ax = fig.add_subplot(111,projection='3d')

ax.scatter(sales['NA_Sales'], sales['Year'],sales['Other_Sales'])

ax.set_xlabel('NA_Sales')

ax.set_ylabel('Year')

ax.set_zlabel('Other_sales')

plt.xticks(rotation=70)

plt.show()
#jointplot

sales.Genre.value_counts().sort_values().head()

sns.jointplot( "Year", "Global_Sales", data=sales)