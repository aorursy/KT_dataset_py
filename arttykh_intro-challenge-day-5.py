import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import scipy.stats as stats



data = pd.read_csv("../input/cereal.csv")

type = data['type'].value_counts()

print(type)

#vitamines = data['vitamines']

print(stats.chisquare(type))

sns.countplot(data['calories']).set_title('Ammount of calories in cereals');
