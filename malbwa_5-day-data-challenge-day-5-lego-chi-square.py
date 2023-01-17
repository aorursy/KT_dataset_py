import matplotlib.pyplot as plt

import numpy as np

import os

import pandas as pd

import seaborn as sns

from scipy.stats import chisquare



# Input data files are available in the "../input/" directory.



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# Create list of data files to read into data structure

data_files = [

    'colors.csv',

    'inventories.csv',

    'inventory_parts.csv',

    'inventory_sets.csv',

    'part_categories.csv',

    'parts.csv',

    'sets.csv',

    'themes.csv'

]



data = {}
# Read csv data files into 'data' dictionary with

# filename as the key as a pandas dataframe

for file in data_files:

    file_path = f'../input/{file}'

    file_nm, file_ext = os.path.splitext(file)

    if file_nm not in data:

        data[file_nm] = pd.read_csv(file_path)
for key in data:

    print(key)
for key in data:

    print('########################################\n')

    print(key, ': ', list(data[key]))

    print('shape : ', data[key].shape, '\n')

    for item in data[key]:

        print(type(data[key][item][0]))

#     print(list(data[key]))
# for key in data:

#     print('########################################\n')

#     print(key, ': ')

#     print(data[key].describe(), '\n')
fig, ax = plt.subplots(figsize=(16,4))

sns.countplot(data['sets']['year']).set_title('Lego Sets by Year')



for label in ax.xaxis.get_ticklabels():

    label.set_visible(False)

for label in ax.xaxis.get_ticklabels()[::3]:

    label.set_visible(True)



plt.xticks(rotation=45)

plt.show()
data['colors'].head()
data['inventory_parts'].head()
parts_color = data['inventory_parts']['color_id']

color_id = data['colors']['id']



part_color_merge = data['inventory_parts'].merge(data['colors'], left_on='color_id', right_on='id', how='left')

part_color_merge = part_color_merge.drop(['id'], axis=1)

part_color_merge.head()
chisquare(part_color_merge['color_id'], f_exp=part_color_merge['inventory_id'])
data['sets'].head()
data['inventories'].head()
data['inventory_sets'].head()
for key in data:

#     print('########################################\n')

    print(key, ': ', list(data[key]))

#     print('shape : ', data[key].shape, '\n')

#     for item in data[key]:

#         print(type(data[key][item][0]))
# print(data['inventories'].head())



check_col = data['inventories'][data['inventories']['id'] == 35]

print(check_col)



# print(data['inventory_sets'].head())



check_col2 = data['inventory_sets'][data['inventory_sets']['inventory_id'] == 35]

print(check_col2)
len(data['sets']['set_num'].unique()), len(data['inventories']['set_num'].unique()), len(data['inventory_sets']['set_num'].unique()), 
sets_inventories_merge = data['sets'].merge(data['inventories'], left_on='set_num', right_on='set_num', how='left')

# part_color_merge = part_color_merge.drop(['id'], axis=1)

sets_inventories_merge.head()
len(sets_inventories_merge['set_num']), len(data['inventory_sets']['set_num'])
parts_merge = data['part_categories'].merge(data['parts'], left_on='id', right_on='part_cat_id', how='left')

parts_merge = parts_merge.drop(['id'], axis=1)

parts_merge.head()
sets_inv_theme_merge = sets_inventories_merge.merge(data['themes'], left_on='theme_id', right_on='id', how='left')

# parts_theme_merge = parts_theme_merge.drop(['id'], axis=1)

sets_inv_theme_merge.head()
sets_merge = sets_inv_theme_merge.merge(data['inventory_sets'], left_on='set_num', right_on='set_num', how='left')

# parts_theme_merge = parts_theme_merge.drop(['id'], axis=1)

sets_merge.head()

lego_parts_merge = part_color_merge.merge(parts_merge, left_on='part_num', right_on='part_num', how='left')

# parts_theme_merge = parts_theme_merge.drop(['id'], axis=1)

lego_parts_merge.head()
###### colors :  ['id', 'name', 'rgb', 'is_trans']

###### inventories :  ['id', 'version', 'set_num']

###### inventory_parts :  ['inventory_id', 'part_num', 'color_id', 'quantity', 'is_spare']

###### inventory_sets :  ['inventory_id', 'set_num', 'quantity']

###### part_categories :  ['id', 'name']

###### parts :  ['part_num', 'name', 'part_cat_id']

###### sets :  ['set_num', 'name', 'year', 'theme_id', 'num_parts']

###### themes :  ['id', 'name', 'parent_id']





###### part_color_merge <== inventory_parts + colors

###### sets_inventories_merge <== inventories + sets

###### parts_merge <== part_categories + parts



###### sets_inv_theme_merge <== themes + sets_inventories_merge



# sets_merge <== inventory_sets + sets_inv_theme_merge

# lego_parts_merge <== part_color_merge + parts_merge
dfs = [

    lego_parts_merge,

    sets_merge

]



for df in dfs:

    print(list(df))

lego_parts_merge.head(3), sets_merge.head(3)
lego__merge = lego_parts_merge.merge(sets_merge, left_on='inventory_id', right_on='inventory_id', how='left')

# parts_theme_merge = parts_theme_merge.drop(['id'], axis=1)

lego__merge.head()
lego__merge.shape
lego_parts_merge.shape, sets_merge.shape
lego__merge.head()