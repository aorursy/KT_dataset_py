# for performing mathematical operations

import numpy as np 



# for data processing, CSV file I/O 

import pandas as pd 



# visualizing inventory_parts that has most colors using matplotlib

import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns
# read the data from the csv files into a dataframe

themes = pd.read_csv('../input/lego-database/themes.csv', index_col=0)

sets = pd.read_csv('../input/lego-database/sets.csv', index_col=0)

parts = pd.read_csv('../input/lego-database/parts.csv', index_col=0)

part_categories = pd.read_csv('../input/lego-database/part_categories.csv', index_col=0)

inventories = pd.read_csv('../input/lego-database/inventories.csv', index_col=0)

inventory_sets = pd.read_csv('../input/lego-database/inventory_sets.csv', index_col=0)

inventory_parts = pd.read_csv('../input/lego-database/inventory_parts.csv', index_col=0)

colors = pd.read_csv('../input/lego-database/colors.csv', index_col=0)
# checking first twenty rows for colors csv file

colors.head(20)
# checking the info of the colors dataset

colors.info()
# checking the shape of the dataset

colors.shape
# checking the number of transparent and non transparent colors

colors['is_trans'].value_counts()
matplotlib.rcParams.update({'font.size': 20})



# visualize transparent vs non transparent colors

transparent = colors['is_trans'] == 't'

non_transparent = colors['is_trans'] == 'f'



# data to plot

labels = 'Transparent Colors', 'Non Transparent Colors'

sizes = [transparent.sum(), non_transparent.sum()]

colors = ['lightcoral', 'lightskyblue']



# explode 1st slice

explode = (0.1, 0) 



fig, axs = plt.subplots(figsize=(14, 7))

plt.pie(sizes, explode=explode, labels=labels, colors=colors,

autopct='%1.1f%%', shadow=True, startangle=90)



plt.axis('equal')

plt.show()
# checking first twenty rows for parts csv file

parts.head(20)
# checking the info of the parts dataset

parts.info()
# checking the shape of the dataset

parts.shape
# checking first twenty rows for part_categories csv file

part_categories.head(20)
# checking the shape of the dataset

part_categories.shape
# creating new dataframe with part_categories and their parts

parts_with_categories = pd.merge(left=part_categories, right=parts, left_on='id', right_on='part_cat_id')

parts_with_categories = parts_with_categories.rename(columns={'name_x': 'Part_Category_Name', 'name_y':'Part_Name'})

parts_with_categories.head(20)
# grouping categories and counting their respective number of parts

parts_with_categories = parts_with_categories['Part_Category_Name'].value_counts()

parts_with_categories.sort_values(ascending=False)
matplotlib.rcParams.update({'font.size': 16})



fig, axs = plt.subplots(figsize=(18,4))

parts_with_categories.plot(kind="bar", color="brown", alpha=0.6, width= 0.8)



plt.ylabel('Number of Parts')

plt.title('Number of Parts in Each Part Category')

plt.xticks(rotation=90)

plt.legend()



plt.show()
# checking first twenty rows for inventories csv file

inventories.head(20)
# checking the info of the parts dataset

inventories.info()
# checking the shape of the dataset

inventories.shape
# grouping each version and counting the frquency of sets in each group of inventory

inventories['version'].value_counts()
sets_per_inventory_parts = inventories['version'].value_counts()

sets_per_inventory_parts.sort_values(ascending=False)
fig, axs = plt.subplots(figsize=(16,4))

sets_per_inventory_parts.plot(kind="bar", color="green", alpha=0.6, width= 0.5)



plt.ylabel('Number of Sets')

plt.title('Inventory Version')

plt.xticks(rotation=0)

plt.grid()



plt.show()
unique_inventory_parts = inventory_parts[['color_id']]

unique_inventory_parts = unique_inventory_parts.groupby('inventory_id').count()



# taking out the top 15 inventory parts with most colors available

inventory_parts_most_colors = unique_inventory_parts.sort_values(by='color_id', ascending=False)

inventory_parts_most_colors = inventory_parts_most_colors[0:15]

inventory_parts_most_colors
matplotlib.rcParams.update({'font.size': 16})



fig, axs = plt.subplots(figsize=(18,9))

inventory_parts_most_colors['color_id'].plot(kind="barh", color="orange", alpha=0.6, width= 0.8)



plt.xlabel('Number of Colors Available')

plt.ylabel('Inventory Ids')

plt.title('Inventory Parts that has Most Colors Available')

plt.xticks(rotation=90)

plt.legend()

plt.grid()

axs.set_xticks(np.arange(0,800,20))



plt.show()
sns.set(style="whitegrid")



# initialize the matplotlib figure

fig, axs = plt.subplots(figsize=(18,9))



# plot the Total Missing Values

sns.set_color_codes("bright")

sns.barplot(x=inventory_parts_most_colors.index, y="color_id", data=inventory_parts_most_colors, color="r")



# customizing Bar Graph

plt.xticks(rotation='90')

plt.xlabel('Inventory Parts', fontsize=15)

plt.ylabel('Number of Colors Available', fontsize=15)

plt.title('Numebr of Colors available per Inventory Part', fontsize=20)