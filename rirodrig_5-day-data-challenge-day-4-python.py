import matplotlib.pyplot as plt

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns # visualization library

sns.set_style("dark")

sns.despine()
data  = pd.read_csv('../input/database.csv')

data.head(1)
plt.figure(figsize=(10,100))

colors = sns.color_palette("muted")

graph = sns.countplot(y = data['Aircraft'], palette=colors)

graph.set_xlabel(xlabel='Numer of Incidents', fontsize=16)

graph.set_ylabel(ylabel='Aircraft', fontsize=16)

graph.set_title(label='Number of Incidents Per Aircraft type', fontsize=20)

plt.show()