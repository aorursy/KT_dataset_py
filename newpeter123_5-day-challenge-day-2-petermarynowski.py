import pandas as pd

import matplotlib.pyplot as plt



nutrition = pd.read_csv('../input/starbucks_drinkMenu_expanded.csv')

nutrition.describe(include='all') # only show numerics or add include='all' for all data

nutrition.columns # first list the columns in case of spaces etc

sodium = nutrition[" Sodium (mg)"]

plt.hist(sodium)

plt.title("Sodum in Starbucks Menu Items")







nutrition.hist(column= " Sodium (mg)", figsize=(12,7), bins = 9)

plt.title("Sodium Stgarbucks")

plt.xlabel("Sodium in milligrams")

plt.ylabel("Count")


