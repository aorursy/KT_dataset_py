import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt



recipes = pd.read_csv('/kaggle/input/animal-crossing-new-horizons-nookplaza-dataset/recipes.csv')

rp = recipes.iloc[:,0:15]

rp.head(5)
def get_materials(item_name):

    # Slice the table for the item we want.

    material_list = rp[rp['Name'] == item_name]

    

    # Initialize dictionary and iterator

    material_dict = {}

    i = 0

    

    # Iterate over the material and quantity columns

    # The material name is our key, the quantity is our value to that key

    while i < 6:

        material_dict[material_list.iloc[0,i*2+2]] = material_list.iloc[0,i*2+1]

        i += 1

    

    # Get rid of the nan value keys since we don't need them.

    if np.nan in material_dict:

        material_dict.pop(np.nan)

    

    # Clean our 'Sell' value so that it is an integer, not an array.

    material_dict['Sell'] = material_list['Sell'].values.sum()

    return material_dict



print(get_materials('tire toy'))

print(get_materials('barrel'))
# Import the data for the raw materials

mat = pd.read_csv('/kaggle/input/animal-crossing-new-horizons-nookplaza-dataset/other.csv')



# Function for calculating difference between sell price and sell price of all materials in recipe.

def materials_diff(item_name):

    # Run the get_materials function to create dictionary

    mat_dict = get_materials(item_name)

    

    # Initialize sell prices

    mat_price = 0

    sell_price = 0

    

    for key, val in mat_dict.items():

        # The sell price becomes the same as sell_price (integer only)

        if key == 'Sell':

            sell_price = val.sum()

        

        # The materials in our dictionary are matched to the Sell prices in our raw materials table.

        # Material prices are multiplied by the quantity (values) from the dictionary to get total raw material sell price.

        else:

            material_cost = mat[mat["Name"] == key]['Sell'].values.sum()

            mat_price = mat_price + material_cost * val

    return sell_price-mat_price



print(materials_diff('tire toy'))

print(materials_diff('barrel'))
# Generate recipe name list

name_list = list(rp.Name)



# Iterate over recipe name list to create a list of craft-to-sell profit margins.

craft_profit = [materials_diff(x) for x in name_list]



# Make this list a column and check table

rp['craft_profit'] = craft_profit

rp.head(5)
# Make a new table only containing profitable recipes.

profitable_recipes = rp[rp['craft_profit'] > 0].sort_values(by=['craft_profit'], ascending=False)

profitable_recipes
sns.distplot(profitable_recipes['craft_profit'], bins=10)

plt.show()
ins = pd.read_csv('/kaggle/input/animal-crossing-new-horizons-nookplaza-dataset/insects.csv')

fish = pd.read_csv('/kaggle/input/animal-crossing-new-horizons-nookplaza-dataset/fish.csv')

sns.distplot(ins['Sell'], bins=10)

sns.distplot(fish['Sell'], bins=10)

plt.xlim(0, 25000)

plt.show()