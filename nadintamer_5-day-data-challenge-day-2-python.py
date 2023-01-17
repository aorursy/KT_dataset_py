#import necessary libraries

import pandas as pd 

import matplotlib.pyplot as plt



#read .csv file of nutrition data

nutrition_data = pd.read_csv("../input/starbucks_drinkMenu_expanded.csv")

#isolate the "Calories" column of nutrition_data

calories = nutrition_data["Calories"]

#plot a histogram of calories data

plt.hist(calories, bins=9, edgecolor = "black")

plt.title("Calories in Starbucks Menu Items") # add a title

plt.xlabel("Calories in kcal") # label the x axes 

plt.ylabel("Count") # label the y axes