#These are all of the dependencies that we will need for the analysis.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#To begin, we will load in the data and create a very basic histogram in order to see how 
#calories are distributed in the dataset. The code will also include a print out of very
#basic summary statistics.
Menu = pd.read_csv('../input/menu.csv')
plt.hist(Menu.Calories)
plt.show()
print('Summary Statistics for Calories: {}'.format(Menu['Calories'].describe()))
#The code below will show the menu item with the highest calorie count on the menu
HighCalorie = Menu.query('Calories > 1500')
print(HighCalorie['Item'].head(5))
#The code below will show the menu item with the lowest calorie count on the menu
LowCalorie = Menu.query('Calories < 10')
print(LowCalorie['Item'].head(5))
#The purpose of the code below is to group all of the foods by category and to plot the
#average calary count of items in the group.

Calories = Menu.drop('Item', axis = 1)
Calories = Menu.groupby(["Category"])["Calories"].mean()
Calories = Calories.sort_values(ascending=False)
print(Calories)

import matplotlib.pyplot as plt

Category = ['Chicken & Fish', 'Smoothies & Shakes', 'Breakfast',
'Beef & Pork', 'Coffee & Tea', 'Salads', 'Snacks & Sides','Desserts','Beverages']

plt.figure(figsize = (20,10))
plt.suptitle('Calories by Category', fontsize = 24)
plt.xlabel('Category', fontsize = 20)
plt.ylabel("Calories", fontsize = 20)
plt.bar(Category, Calories)
plt.show()
#I was suprised by the amount of average calaries that were in the "Smoothies & Shakes"
#the code below takes a look at the highest Calorie counts in that category.
SmoothShakes = Menu[Menu["Category"] == "Smoothies & Shakes"]
SmoothShakes = SmoothShakes.sort_values("Calories", ascending = False)
print(SmoothShakes[["Item", "Calories"]].head(5))