import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
Wine = pd.read_csv('../input/winemag-data-130k-v2.csv')
Wine.head()
Wine.tail()
Wine.info()
Wine.describe()
Duplicates_Wines = Wine.groupby(['title', 'taster_name', 'description'])

# we´re counting how often the aggregation occurs. Therefore we can use any kind of column

Duplicates_Wines['points'].count().reset_index().sort_values(by = 'points', ascending=False).head(n = 10)
# Lets take an example to review our code if there was any error:

Wine[Wine['title'] == 'Benmarl 2014 Slate Hill Red (New York)']
# Create new Frame for Duplicates



WineDuplicates = Duplicates_Wines['points'].count().reset_index().sort_values(by = 'points', ascending=False)

WineDuplicates = WineDuplicates[WineDuplicates['points'] > 1]

# review output of the lesser counts in dataframe it should be not less than 2

WineDuplicates.tail()
# drop duplicates from main dataframe

Wine.drop_duplicates(subset = ['title', 'taster_name', 'description'], inplace = True)
len(Wine.index)
len(Wine["country"].unique()) 
Wine["country"].value_counts().head(n = 10)
sns.countplot(x = "country", data = Wine, order = Wine["country"].value_counts().iloc[:9].index)

plt.title('Countries with the most wine representations')

plt.tight_layout()
Wine_Country_Grouping = Wine.groupby('country')

Wine_Country_Grouping_List = Wine_Country_Grouping['points'].mean().reset_index()

# sort Values and get only top 10

Wine_Country_Grouping_List = Wine_Country_Grouping_List.sort_values(by = 'points', ascending = False).iloc[:10]

# while testing - the results are kind of similar - a visualization does not really help at this point

Wine_Country_Grouping_List

len(Wine["taster_name"].unique()) 
top_ten_taster = Wine["taster_name"].value_counts().iloc[:10].reset_index().rename(columns = {'index' : 'Taster', 'taster_name': 'count'})

top_ten_taster
print('How many percentage of representations do have the top 10 taster?: {}'.format( sum(top_ten_taster['count']) / len(Wine.index) ))
Wine_Taster = pd.DataFrame(columns = Wine.columns)

for taster in range(0, len(top_ten_taster.index)):

      bool_taster = top_ten_taster['Taster'][taster] == Wine['taster_name']

      Wine_Taster = Wine_Taster.append(Wine[bool_taster == True], ignore_index= True)
Wine_Taster.info()
len(Wine_Taster['title'].unique())
price_not_null = Wine_Taster['price'].notnull()

Wine_Taster_Prices = Wine_Taster[price_not_null == True]

Wine_Taster_Grouped = Wine_Taster_Prices.groupby('taster_name')
Wine_Taster_Points = pd.DataFrame()

Wine_Taster_Points['minPoints'] = Wine_Taster_Grouped['points'].min()

Wine_Taster_Points['maxPoints'] = Wine_Taster_Grouped['points'].max()

Wine_Taster_Points
Wine_Taster_Grouped['price'].describe()
sns.countplot(x = "country", data = Wine_Taster, order = Wine_Taster["country"].value_counts().iloc[:9].index)

plt.title('Countries with the most wine representations')

plt.tight_layout()
Wine_Taster_Points_Grouped = Wine_Taster.groupby('points')

sns.barplot(x = 'points', y = 'price', data = Wine_Taster_Points_Grouped['price'].mean().reset_index())

plt.tight_layout()
sns.barplot(x = 'points', y = 'price', data = Wine_Taster_Points_Grouped['price'].min().reset_index())

plt.tight_layout()
sns.barplot(x = 'points', y = 'price', data = Wine_Taster_Points_Grouped['price'].max().reset_index())

plt.tight_layout()
# Get an Overview of each Taster by himself

# To form such a Facet_Grid I used the example from: 

### http://seaborn.pydata.org/examples/many_facets.html ###

Wine_Taster_PT_Grouped = Wine_Taster.groupby(['taster_name', 'points'])

grid = sns.FacetGrid(Wine_Taster_PT_Grouped['price'].mean().reset_index(), 

                     col="taster_name", hue="taster_name", 

                     col_wrap=4, size=3)

grid.map(plt.plot, "points", "price")

grid.fig.tight_layout(w_pad=1.5)