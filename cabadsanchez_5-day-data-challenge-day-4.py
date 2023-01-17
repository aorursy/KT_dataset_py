import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns



cereals = pd.read_csv('../input/cereal.csv')
sns.countplot(y = 'mfr', data = cereals).set_title('Manufacturer of cereal')
# bubble chart

import matplotlib.pyplot as plt

x = cereals['sugars']

y = cereals['calories']

z = cereals['rating']



plt.scatter(x, y, s=z, c="green", alpha=0.4, linewidth=6)

plt.xlabel("Grams of sugars")

plt.ylabel("Calories per serving")

plt.title("Bubble chart", loc="right")