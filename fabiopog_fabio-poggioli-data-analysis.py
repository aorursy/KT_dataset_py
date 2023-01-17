import pandas as pnd

data = pnd.read_csv("/kaggle/input/starbucks-menu-nutrition-drinks.csv")

data.head()
import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



data['Calories'].plot(kind='hist')

plt.title('Histogram of Calories')

plt.xlabel('Calories')
sns.set(style="ticks")

sns.boxplot(x=data['Brand'], y=data['Calories']);
sns.boxplot(x=data['Kind'], y=data['Calories'], data=data)
im = sns.FacetGrid(data, col="Brand", hue='Calories')

im.map(sns.swarmplot, 'Kind', 'Calories', alpha=1)

im.add_legend();