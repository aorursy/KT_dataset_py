import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import re

%matplotlib inline

#Read and transform the data

data = pd.read_csv('../input/menu.csv')

print(data.dtypes)



# Each category

categories = np.array(data.Category.unique())

print(categories)

breakfastMeals = data[data.Category == 'Breakfast']

beef_porkMeals = data[data.Category == 'Beef & Pork']

chicken_fishMeals = data[data.Category == 'Chicken & Fish']

saladMeals = data[data.Category == 'Salads']

snacks_sidesMeals = data[data.Category == 'Snacks & Sides']

desserts = data[data.Category == 'Desserts']

beverages = data[data.Category == 'Beverages']

coffee_tea = data[data.Category == 'Coffee & Tea']

smoothies_shakes = data[data.Category == 'Smoothies & Shakes']

# print(breakfastMeals.shape[0]) # Number of breakfast meals...

# print(beef_porkMeals.shape[0]) # Number of beef and pork meals...

# print(chicken_fishMeals.shape[0]) # Number of chicken and fish meals

# print(saladMeals.shape[0]) # Number of salad meals

# print(snacks_sidesMeals.shape[0]) # Number of snacks and sides 

# print(desserts.shape[0]) # Number of desserts

# print(beverages.shape[0]) # Number of beverages

# print(coffee_tea.shape[0]) # Number of coffee and tea

# print(smoothies_shakes.shape[0]) # Number of smoothies
# Which meal takes up a great amount of the total meals available in the menu?

amtPerCategory = np.array([breakfastMeals.shape[0], 

                           beef_porkMeals.shape[0], 

                           chicken_fishMeals.shape[0],

                           saladMeals.shape[0],

                           snacks_sidesMeals.shape[0],

                           desserts.shape[0],

                           beverages.shape[0],

                           coffee_tea.shape[0],

                           smoothies_shakes.shape[0]

                          ])

colors = np.array(['c', '#94edc0', '#c238a7', '#b21839', '#432f13', '#7090fa', '#15e925',

                  '#dff970', '#d4a661'])

plt.pie(amtPerCategory, labels=categories, colors=colors,startangle=90,shadow=True,

       explode=(0,0,0,0,0,0,0,0.15,0), autopct='%1.2f',radius = 1.75)

plt.title("Meals per Category")

plt.gca().set_aspect('equal')
# How much calories does an average meal per category contribute to a male's daily caloric intake?

# Avg. Calorie according to Meg Campbell: https://www.livestrong.com/article/457078-normal-caloric-intake-for-men/

avgBreakfastCal = breakfastMeals.Calories.mean()

avgBeefPorkCal = beef_porkMeals.Calories.mean()

avgChickenFishCal = chicken_fishMeals.Calories.mean()

avgSaladCal = saladMeals.Calories.mean()

avgSnacksSidesCal = snacks_sidesMeals.Calories.mean()

avgDessertsCal = desserts.Calories.mean()

avgBeveragesCal = beverages.Calories.mean()

avgCoffeeTeaCal = coffee_tea.Calories.mean()

avgSmoothiesShakesCal = smoothies_shakes.Calories.mean()

avgCalsPerCategory = np.array([avgBreakfastCal, 

                              avgBeefPorkCal,

                              avgChickenFishCal,

                              avgSaladCal,

                              avgSnacksSidesCal,

                              avgDessertsCal,

                              avgBeveragesCal,

                              avgCoffeeTeaCal,

                              avgSmoothiesShakesCal])

print(avgCalsPerCategory)

np.ndarray.sort(avgCalsPerCategory, kind='heapsort')

y = np.linspace(0, 8, 9)

plt.barh(y, avgCalsPerCategory,height = 0.95, label = categories, color = colors)

plt.yticks(y, categories)

plt.xlabel('Calories')
# Which meal has the most/least amount of sodium?

# Particularly helpful for people with Meniere's Disease as

# less sodium intake is usually prescribed as a way to prevent

# Dizziness and Vertigo

# 

avgBreakfastSodium = breakfastMeals.Sodium.mean()

avgBeefPorkSodium = beef_porkMeals.Sodium.mean()

avgChickenFishSodium = chicken_fishMeals.Sodium.mean()

avgSaladSodium = saladMeals.Sodium.mean()

avgSnacksSidesSodium = snacks_sidesMeals.Sodium.mean()

avgDessertsSodium = desserts.Sodium.mean()

avgBeveragesSodium = beverages.Sodium.mean()

avgCoffeeTeaSodium = coffee_tea.Sodium.mean()

avgSmoothiesShakesSodium = smoothies_shakes.Sodium.mean()

avgSodiumPerCategory = np.array([avgBreakfastSodium, 

                              avgBeefPorkSodium,

                              avgChickenFishSodium,

                              avgSaladSodium,

                              avgSnacksSidesSodium,

                              avgDessertsSodium,

                              avgBeveragesSodium,

                              avgCoffeeTeaSodium,

                              avgSmoothiesShakesSodium])

np.ndarray.sort(avgSodiumPerCategory, kind='mergesort')

print(avgSodiumPerCategory)

y = np.linspace(0, 8, 9)

plt.barh(y, avgSodiumPerCategory,height = 0.95, label = categories, color = colors)

plt.yticks(y, categories)

plt.xlabel('Sodium')