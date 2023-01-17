import matplotlib.pyplot as plt

import pandas as pd
nutrition = pd.read_csv('../input/starbucks_drinkMenu_expanded.csv')
nutrition.describe()

nutrition.describe(include = 'all')
#To figure out the column name

nutrition.columns

#plt.hist()
plt.hist(nutrition[' Sodium (mg)'])

plt.title('Sodium content')
nutrition.hist(column = ' Sodium (mg)', figsize = (10, 5), bins = 30, edgecolor = 'black')