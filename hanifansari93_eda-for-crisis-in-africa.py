import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



#Import the dataset

data = pd.read_csv('/kaggle/input/africa-economic-banking-and-systemic-crisis-data/african_crises.csv')
#Let's have a look at currency rate

sns.set_style('whitegrid')

plt.figure(figsize = (8,5))

sns.lineplot(x = 'year', y = 'exch_usd', hue = 'country', data = data, palette = 'colorblind')

plt.xlabel('Year')

plt.ylabel('Exchange Rate')

display()
#Exchange rates before and after independece

sns.set_style('whitegrid')

fig, axes = plt.subplots(ncols=4, nrows=4, figsize=(18,12), dpi= 60)

axes = axes.flatten()

for i, ax in zip(data['country'].unique(), axes):

  sns.lineplot(x = 'year', y = 'exch_usd', hue = 'independence', data = data[data['country'] == i], ax = ax)

  ax.set_xlabel('Year')

  ax.set_ylabel('Exchange Rate')

  ax.set_title('{}'.format(i))

  ax.get_legend().remove()

handles, labels = ax.get_legend_handles_labels()

fig.legend(handles, labels, loc=1)

fig.subplots_adjust(top=0.95)

for i in range(13,16):

  fig.delaxes(axes[i])

plt.tight_layout()
#Hyperinflation

#Relationship between Inflation annual and Inflation crisis

sns.set_style('whitegrid')

fig, axes = plt.subplots(ncols=4, nrows=4, figsize=(18,12), dpi= 60)

axes = axes.flatten()

for i, ax in zip(data['country'].unique(), axes):

  sns.lineplot(x = 'year', y = 'inflation_annual_cpi', data = data[data['country'] == i], ax = ax, color = 'cornflowerblue')

  ax.set_xlabel('Year')

  ax.set_ylabel('Inflation Rate')

  ax.set_title('{}'.format(i))

  inflation = data[(data['country'] == i) & (data['inflation_crises'] == 1)]['year'].unique()

  for i in inflation:

    ax.axvline(x=i, color='indianred', linestyle='--', linewidth=.9)

fig.subplots_adjust(top=0.95)

for i in range(13,16):

  fig.delaxes(axes[i])

plt.tight_layout()
#Number of inflation crisis by Country

data.groupby('country').agg({'inflation_crises':'sum'}).sort_values('inflation_crises', ascending = False)
#Let's have a look at exchange rate and currency crisis

sns.set_style('whitegrid')

fig, axes = plt.subplots(ncols=4, nrows=4, figsize=(18,12), dpi= 60)

axes = axes.flatten()

for i, ax in zip(data['country'].unique(), axes):

  sns.lineplot(x = 'year', y = 'exch_usd', data = data[data['country'] == i], ax = ax, color = 'mediumslateblue')

  ax.set_xlabel('Year')

  ax.set_ylabel('Exchange Rate')

  ax.set_title('{}'.format(i))

  currency = data[(data['country'] == i) & (data['currency_crises'] == 1)]['year'].unique()

  for i in currency:

    ax.axvline(x=i, color='indianred', linestyle='--', linewidth=.9)

fig.subplots_adjust(top=0.95)

for i in range(13,16):

  fig.delaxes(axes[i])

plt.tight_layout()

display()
#Number of inflation crisis by Country

data.groupby('country').agg({'currency_crises':'sum'}).sort_values('currency_crises', ascending = False)
#Relationship between Exchange Rate and Inflation rate

fig, axes = plt.subplots(ncols=4, nrows=4, figsize=(18,12), dpi= 60)

axes = axes.flatten()

for i, ax in zip(data['country'].unique(), axes):

  sns.lineplot(x = 'year', y = 'exch_usd', data = data[data['country'] == i], ax = ax, color = 'indianred', label = 'Exchange Rate')

  ax2 = ax.twinx()

  sns.lineplot(x = 'year', y = 'inflation_annual_cpi', data = data[data['country'] == i], ax = ax2, color = 'slateblue', label = 'Inflation Rate')

  ax.set_xlabel('Year')

  ax.set_ylabel('Exchange Rate')

  ax.get_legend().remove()

  ax2.set_ylabel('Inflation Rate')

  ax2.get_legend().remove()

  ax.set_title('{}'.format(i))

handles, labels = ax.get_legend_handles_labels()

handles2, labels2 = ax2.get_legend_handles_labels()

fig.legend(handles + handles2, labels + labels2, loc=1)

fig.subplots_adjust(top=0.95)

for i in range(13,16):

  fig.delaxes(axes[i])

plt.tight_layout()

display()
#Mapping the values in banking_crisis to 0 and 1

dict = {'no_crisis': 0, 'crisis': 1}

data['banking_crisis'] = data['banking_crisis'].map(dict)
#Visualizing different types of crisis

fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(18,12), dpi= 60)

axes = axes.flatten()

cols = ['currency_crises','inflation_crises','banking_crisis','systemic_crisis']

for i, ax in zip(cols, axes):

  sns.countplot(y = 'country', ax = ax, data = data, hue = i, palette = 'Paired')

plt.tight_layout()

display()
#Visualizing different types of debts

fig, axes = plt.subplots(ncols=1, nrows=2, figsize=(18,7), dpi= 60)

axes = axes.flatten()

cols = ['domestic_debt_in_default','sovereign_external_debt_default']

for i, ax in zip(cols, axes):

  sns.countplot(x = 'country', ax = ax, data = data, hue = i)

plt.tight_layout()

display()