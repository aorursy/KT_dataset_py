#required modules

import pandas as pd

import seaborn as sns

import numpy as np

import matplotlib.pyplot as plt

from matplotlib.ticker import PercentFormatter



# Reading the data

nobel = pd.read_csv("../input/archive.csv")



# The first few winners

nobel.head(10)
print("Total (some shared) prizes:",len(nobel))

print("\nPrizes by gender and country:")

display(nobel['Sex'].value_counts())

display(nobel['Birth Country'].value_counts().head(10))
# Proportion of US-born winners per decade

nobel['USA-born Winners'] = nobel['Birth Country'] == 'United States of America'

nobel['Decade'] = (np.floor(nobel['Year'] / 10) * 10).astype('int64')

prop_usa_winners = nobel[['USA-born Winners', 'Decade']].groupby('Decade', as_index = False).mean()



print("Proportion of winners born in the USA, per decade:")

display(prop_usa_winners)



sns.set_style("whitegrid")

plt.rcParams['figure.figsize'] = [12, 7]

ax = sns.lineplot(x = "Decade", y = "USA-born Winners", data = prop_usa_winners)

ax.yaxis.set_major_formatter(PercentFormatter())
# Proportion of female laureates per decade

nobel['Female Winners'] = nobel['Sex'] == 'Female'

prop_female_winners = nobel[['Female Winners', 'Decade', 'Category']].groupby(['Decade', 'Category'], as_index = False).mean()



ax = sns.lineplot(x = "Decade", y = "Female Winners", hue = "Category", data = prop_female_winners)

ax.yaxis.set_major_formatter(PercentFormatter())
nobel[nobel['Sex'] == "Female"].nsmallest(1, 'Year')
nobel['Birth Date'] = pd.to_datetime(nobel['Birth Date'], errors = 'coerce')

nobel['Age'] = nobel['Year'] - nobel['Birth Date'].dt.year



ax = sns.lmplot(x = 'Year', y = 'Age', data = nobel, lowess = True, aspect = 2, line_kws = {'color' : 'black'})
ax = sns.lmplot(x = 'Year', y = 'Age', data = nobel, row = 'Category', lowess = True, aspect = 2, line_kws = {'color' : 'black'})
print(nobel.nsmallest(1, 'Age')['Full Name'])