# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from collections import Counter

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#import training cooking data
cooking_data_train = pd.read_json('../input/train.json').set_index('id')
cooking_data_test = pd.read_json('../input/test.json').set_index('id')
cooking_data = cooking_data_train.append(cooking_data_test, sort=True)
#information about the first entries of the dataset
cooking_data_train.info()
cooking_data_test.info()
cooking_data.info()
#function to find unique values in lists from pandas series rows
#data = series with lists in its rows
#column_name = name of the column in the result dataframe
def diff_ingredients(data, column_name):
    lista_ingredients = [item for ingredient_list in data for item in ingredient_list]
    lista_ingredients = pd.DataFrame(pd.Series(sorted(lista_ingredients)).unique(), columns=[column_name])
    return lista_ingredients
#compare the ingredients from train and test datasets
ingredients_train = diff_ingredients(cooking_data_train.ingredients, 'ingredients')
ingredients_test = diff_ingredients(cooking_data_test.ingredients, 'ingredients')

ingredients_unique_test = ingredients_test.loc[~ingredients_test.ingredients.isin(ingredients_train.ingredients)]
#432 ingredients are in test dataset but not in the train dataset
ingredients_unique_test.info()
#find which ingredients are unique to test dataset
count_test_unique_ingredients = Counter([item for ingredient_list in cooking_data_test.ingredients for item in ingredient_list if (item == ingredients_unique_test.ingredients).any()])
#there are a lot of unique ingredients, but their do not have high frequency
print(count_test_unique_ingredients.most_common(10))
#show frequency of cuisine
cooking_group_cuisine = cooking_data_train.groupby('cuisine').count().sort_values(by='ingredients', ascending=True)

#plot a graph for cuisine frequency
cooking_group_cuisine.plot(kind='barh', width=0.9, title='Cuisine frequency by origin')
plt.gcf().set_size_inches(16,10)
#Summary of ingredients quantity by origin of the dish function
def summary_info(data, bin_number):
    quantity_ingredients = data.str.len()
    print('Median = {}'.format(quantity_ingredients.median())) 
    print('Mean = {:.3}'.format(quantity_ingredients.mean()))
    print('Max = {}'.format(quantity_ingredients.max()))
    print('Min = {}'.format(quantity_ingredients.min()))

    #bar graph of frequency by quantity of ingredients
    plt.hist(quantity_ingredients, bins = bin_number);
    plt.gcf().set_size_inches(16,10)
#Summary of ingredients quantity by italian dish
cooking_italian = cooking_data_train.loc[cooking_data_train.cuisine == 'italian']
cooking_italian.head()
summary_info(cooking_italian.ingredients, 65)
#Summary of ingredients quantity by mexican dish
cooking_mexican = cooking_data_train.loc[cooking_data_train.cuisine == 'mexican']
cooking_mexican.head()
summary_info(cooking_mexican.ingredients, 50)
#Summary of ingredients quantity by southern_us dish
cooking_southern_us = cooking_data_train.loc[cooking_data_train.cuisine == 'southern_us']
cooking_southern_us.head()
summary_info(cooking_southern_us.ingredients, 40)
#Summary of ingredients quantity by indian dish
cooking_indian = cooking_data_train.loc[cooking_data_train.cuisine == 'indian']
cooking_indian.head()
summary_info(cooking_indian.ingredients, 50)