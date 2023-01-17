import matplotlib.pyplot as plt

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns



bites = pd.read_csv('../input/Health_AnimalBites.csv')

bites.head()
# bar plot of all recorded bites per species

ax = sns.countplot(x = 'SpeciesIDDesc', data=bites)

ax.set_title('Bites per Species')

plt.xticks(rotation=45)
# pie chart of bite location

bite_location = bites.groupby('WhereBittenIDDesc').size()

labels = list(bite_location.index)

sizes = bite_location.values / bite_location.values.sum()

colors = ['gold', 'yellowgreen', 'lightskyblue']

plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, shadow=True)

plt.axis('equal')