import numpy as np

import pandas as pd

from scipy.stats import chisquare



import matplotlib.pyplot as plt

import seaborn as sns

#import gmplot
pizza = pd.read_csv("../input/8358_1.csv")

print("Data open : {} columns and {} lines"\

      .format(len(pizza.columns), len(pizza)))

print("Columns are : \n{}".format(pizza.columns))

print("Percentage of absent data : \n{}".format(pizza.isnull().sum()/len(pizza)*100))
diffPizza = pizza['menus.name'].value_counts()

print("Different pizzas : {}".format(diffPizza.count()))
print("Top 10 : \n{}".format(diffPizza[:10]))

diffPizza[:10].plot.pie()

plt.show()
diffRestaurant = pizza['id'].value_counts()

print("Different restaurants : {}".format(diffPizza.count()))
gmap = gmplot.GoogleMapPlotter(37.428, -122.145, 16)

gmap.scatter(pizza['latitude'], pizza['longitude'])

gmap.draw("Map.html")
# Selection of the first line of each pizzeria

first = pizza.groupby('id').first()

first["province"].value_counts()[:10]
pizzeriaByCity = first["city"].value_counts()

pizzeriaByCity[:10]
bigCities = pizzeriaByCity >= 5

pizzeriaBigCities = first.select(lambda x: bigCities[first.loc[x].city])

pizzaBigCities = pizza.select(lambda x: bigCities[pizza.loc[x].city])



# Verification number of pizza restaurants extracted

assert(len(pizzaBigCities) == bigCities[pizza.city].sum())
print("Cheapest cities :")

# Classify by id and then take price max

maxPrice = pizzaBigCities.groupby('id')["menus.amountMax", "city"].max()

maxPrice.groupby("city")["menus.amountMax"].agg(['mean', 'std', 'count']).sort_values(["mean", "std"])[:10]
print("Most expensive cities :")

maxPrice = pizzaBigCities.groupby('id')["menus.amountMin", "city"].min()

maxPrice.groupby("city")["menus.amountMin"].agg(['mean', 'std', 'count']).sort_values(["mean", "std"], ascending = False)[:10]
diffPizza[:3]
for pizzaName in diffPizza.index[:3]:

    # Selection pizza

    selectionPizza = pizza.groupby("menus.name")["menus.amountMin", "menus.amountMax", "city"].get_group(pizzaName)

    # Group by city and compute mean (mean 1 => compute mean of min and max)

    meanPizza = selectionPizza.groupby("city").mean().mean(1)

    # Sort and drop nan

    meanPizza = meanPizza.sort_values(ascending=False).dropna()

    

    # Plot min and max cities

    plt.figure()

    meanPizza = pd.concat([meanPizza[:10], meanPizza[-10:]])

    meanPizza.plot.barh(title=pizzaName, xlim=(0,25))



    plt.show()