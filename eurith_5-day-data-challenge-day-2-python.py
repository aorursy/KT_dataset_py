import pandas as pd

import matplotlib.pyplot as plt

df = pd.read_csv("../input/starbucks_drinkMenu_expanded.csv")
df.describe()
df.describe(include='all')
print(df.columns)
sugars = df[" Sugars (g)"]

sugars.head()
plt.hist(sugars)

plt.title("Sugars in Starbucks Menu Items")
plt.hist(sugars, bins=20, edgecolor='black')

plt.title("Sugars in Starbucks Menu Items")

plt.xlabel("Sugars in grams")

plt.ylabel("Count")
df.hist(column=" Sugars (g)", figsize=(9,9), bins=20, facecolor='purple', edgecolor='red')