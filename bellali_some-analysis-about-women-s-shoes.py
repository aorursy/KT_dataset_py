import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv('../input/Datafiniti_Womens_Shoes.csv')

df.head()
df.shape
# check out how many null data there

df.info()
# view the DataFrame to find out if there are duplicated data

df.duplicated().sum()
# Check out the unique data of some columns

df['brand'].unique()
df['categories'].unique()
df['primaryCategories'].unique()
df['prices.currency'].unique()
df['prices.isSale'].unique()
# Returns valid descriptive statistics for each column of data

df.describe()
# copy the data

data = df.copy()
# select useful columns to biuld up a new DataFrame

columns_name = ['brand', 'categories', 'name', 'prices.amountMax', 'prices.amountMin', 'prices.color', 'prices.isSale']

data = data[columns_name]

data.head()
# rename "prices.-" columns

data.rename(columns = lambda x: x.strip().lower().replace(".", "_"), inplace=True)
# change brand name to lower case

data['brand'] = data['brand'].str.lower()
# split "categories" to simpler form

def categoriesReplacement(data):

    # Sandals

    data.loc[data['categories'].str.contains('Sandals'), 'categories'] = 'Sandals'

    # Baby & Kid

    data.loc[data['categories'].str.contains('Baby|Kids'), 'categories'] = "Baby&Kids"

    # Slippers

    data.loc[data['categories'].str.contains('Slippers'), 'categories'] = "Slippers"

    # Boots

    data.loc[data['categories'].str.contains('Boots'), 'categories'] = "Boots"

    # Home Shoes

    data.loc[data['categories'].str.contains('Home'), 'categories'] = "Home Shoes"

    # Sports Shoes

    data.loc[data['categories'].str.contains('Sports|Athletic|Running'), 'categories'] = "Sports Shoes"

    # Boat

    data.loc[data['categories'].str.contains('Boat'), 'categories'] = "Boat"

    # Oxford

    data.loc[data['categories'].str.contains('Oxford'), 'categories'] = "Oxford"

    # Dress Shoes

    data.loc[data['categories'].str.contains('Dress Shoes'), 'categories'] = "Dress Shoes"

    # Flats

    data.loc[data['categories'].str.contains('Flats'), 'categories'] = "Flats"

    # Loafers

    data.loc[data['categories'].str.contains('Loafers'), 'categories'] = "Loafers"

    # Clogs

    data.loc[data['categories'].str.contains('Clogs'), 'categories'] = "Clogs"

    # Pumps

    data.loc[data['categories'].str.contains('Pumps'), 'categories'] = "Pumps"
categoriesReplacement(data)
# list the unique data again to find out the rest unclear categories

data['categories'].unique()
# classify the rest categories

data.loc[data['categories'].str.contains("Casual Shoes|Women's Shoes|Womens,Rampage")]
data.loc[data['name'] == 'MUK LUKS Womens Jane Suede Moccasin', 'categories'] = "Loafers"

data.loc[data['name'].str.contains('Boot|Bootie|Boots|Booties'), 'categories'] = 'Boots'

data.loc[data['name'].str.contains('Sandals|Sandal'), 'categories'] = 'Sandals'

data.loc[data['name'].str.contains('Athletic|Canvas|Walk|Walking|New Balance'), 'categories'] = 'Sports Shoes'

data.loc[data['name'].str.contains('Pumps'), 'categories'] = 'Pumps'

data.loc[data['name'].str.contains('Flat|Flats'), 'categories'] = 'Flats'

data.loc[data['name'].str.contains('Clog'), 'categories'] = 'Clogs'
data.loc[data['name'].str.contains('Bear Paw Plush'), 'categories'] = 'Home Shoes'
# recheck

data['categories'].unique()
# combine "prices_amountmax" and "prices_amountmin" to one column

data['prices'] = (data['prices_amountmax'] + data['prices_amountmin']) / 2
data.head()
# select brands with more than 80 pieces data

brand = data['brand'].value_counts()[data['brand'].value_counts() > 80]

brand_names = list(pd.DataFrame(brand).index)
data_brand = data[data['brand'].isin(brand_names)]
data_brand_ave = data_brand.groupby('brand')['prices'].mean().sort_values(ascending=True)

data_brand_ave
fig = plt.figure(figsize=(8,6))

data_brand_ave.plot(kind='barh', color='r', align='center', alpha=.8)

plt.title('Average Price of Each Distinct Brand', fontsize=18)

plt.xlabel('Average Price($)')

plt.ylabel('Brand');
fig = plt.figure(figsize=(8,6))

# draw boxplot for distinct category of shoes

sns.boxplot(y='categories', x='prices', data = data)

plt.title('Price Distribution of Each Distinct Categories', fontsize=18)

plt.xlabel('Price($)')

plt.ylabel('Categories');
# select sports shoes brands from data

data_sports = data_brand.query('categories == "Sports Shoes"')[['brand', 'prices']]

data_sports.head()
sports_shoes_names = list(pd.DataFrame(data_sports['brand'].value_counts()[data_sports['brand'].value_counts() > 91]).index)

sports_shoes = data_sports[data_sports['brand'].isin(sports_shoes_names)]
sns.FacetGrid(sports_shoes,col='brand', hue='brand').map(sns.distplot,"prices").add_legend()

plt.show();
# select boots shoes brands from data

data_boots = data_brand.query('categories == "Boots"')[['brand', 'prices']]

data_boots.head()
boots_shoes_names = list(pd.DataFrame(data_boots['brand'].value_counts()[data_boots['brand'].value_counts() > 100]).index)

boots_shoes = data_boots[data_boots['brand'].isin(boots_shoes_names)]
sns.FacetGrid(boots_shoes,col='brand', hue='brand').map(sns.distplot,"prices").add_legend()

plt.show();
# select the brands we want to explore

brand_names = ['skechers', 'adidas', 'nike', 'new balance']

data_color = data[data['brand'].isin(brand_names)][['brand', 'categories', 'prices', 'prices_color']]

data_color = data_color.query('categories == "Sports Shoes"')

data_color.head()
# choose main color

color = data_color.pivot_table('prices', index='brand', columns='prices_color', aggfunc='count').dropna(axis=1, how='any')

color
data_main_color = data_color[data_color['prices_color'].isin(['Black', 'Black White', 'Gray', 'White'])]

data_main_color_ave = data_main_color.pivot_table('prices', index='brand', columns='prices_color', aggfunc='mean')

data_main_color_ave
fig = plt.figure(figsize=(6,4))

sns.set(style="darkgrid")

x_data = data_main_color_ave.index

y_data1 = data_main_color_ave['Black']

y_data2 = data_main_color_ave['Black White']

y_data3 = data_main_color_ave['Gray']

y_data4 = data_main_color_ave['White']

plt.plot(x_data, y_data1, color='black', linewidth = 3.0, linestyle = '-')

plt.plot(x_data, y_data2, color='silver', linewidth = 3.0, linestyle = '-.')

plt.plot(x_data, y_data3, color='gray', linewidth = 3.0, linestyle = '-')

plt.plot(x_data, y_data4, color='white', linewidth = 3.0, linestyle = '-')

plt.legend()

plt.title("Sports Shoes Average Prices in Diffenrent Colors", fontsize=14)

plt.xlabel('Brands')

plt.ylabel('Price($)');