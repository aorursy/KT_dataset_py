# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
pizza_data = pd.read_csv('/kaggle/input/pizza-restaurants-and-the-pizza-they-sell/Datafiniti_Pizza_Restaurants_and_the_Pizza_They_Sell_May19.csv')
# show 5 head data



pizza_data.head(5)
# get the all column 



print(pizza_data.columns)

print("Number of column : {}".format(len(pizza_data.columns)))
#describe the amount of menu



menupizza = pizza_data[['menus.amountMax', 'menus.amountMin']]

menupizza.describe()

#describe the price of menu

pricepizza = pizza_data[['priceRangeMax', 'priceRangeMin']]

pricepizza.describe()
#Most & Least categories in cities



pizza_data.primaryCategories.value_counts()
# Series to dataframe



categorydata = pizza_data['primaryCategories'].value_counts().reset_index()

categorydata.columns = ['categories', 'values']



categorydata
import matplotlib.mlab as mlab

import matplotlib.pyplot as plt

import seaborn
y = seaborn.barplot(x = categorydata['values'],y=categorydata['categories'])

for index, value in enumerate(categorydata['values']):

    y.text(value , index + .25, str(value), fontweight='bold')
citydata = pizza_data['city'].value_counts().reset_index()

citydata.columns = ['city', 'values']



citydata
#city have more than 100 restaurants



most_citydata = citydata[citydata['values'] > 100]

most_citydata
y = seaborn.barplot(x = most_citydata['values'], y=most_citydata['city'], palette="rocket")

for index, value in enumerate(most_citydata['values']):

    y.text(value , index + .25, str(value), fontweight='bold')
# study the pizza name



menudata = pizza_data['menus.name'].value_counts().reset_index()

menudata.columns = ['menu_name', 'values']



menudata
most_menudata = menudata.head(10)
y = seaborn.barplot(x = most_menudata['values'], y=most_menudata['menu_name'], palette="rocket")

for index, value in enumerate(most_menudata['values']):

    y.text(value , index + .15, str(value), fontweight='bold')