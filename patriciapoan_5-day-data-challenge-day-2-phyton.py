import matplotlib.pyplot as plt

import pandas as pd
nutrition = pd.read_csv("../input/starbucks_drinkMenu_expanded.csv")

nutrition.describe()
# list all the columns name

print(nutrition.columns)

# get the sodium column

sodium = nutrition[" Sodium (mg)"]



# plot histogram of sodium contain

plt.hist(sodium)

plt.title("Sodium in Starbucks Menu Item")
plt.hist(sodium, bins = 10, edgecolor = "yellow")

plt.title("Sodium in Starbucks Menu Item")

plt.xlabel("Sodium in mg")

plt.ylabel("Count")
### another way of plotting a histogram (from the pandas plotting API)

# figsize is an argument to make it bigger

nutrition.hist(column= " Sodium (mg)", figsize = (10,10))