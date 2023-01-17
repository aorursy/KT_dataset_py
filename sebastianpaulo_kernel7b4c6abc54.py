import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from pandas.plotting import register_matplotlib_converters

%matplotlib inline



import warnings

warnings.filterwarnings("ignore")



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Create DataFrame from national_convention.csv

# Bring some order into the columns and format date columns to datetime

cols = ['name', 'start_mandate', 'end_mandate', 'group', 'department', 'date_of_birth','place_of_birth',\

        'date_of_death','place_of_death']

date_cols = ['start_mandate', 'end_mandate', 'date_of_birth', 'date_of_death']

df = pd.read_csv('/kaggle/input/deputies-of-the-french-national-convention/national_convention.csv')

for col in date_cols:

    df[col] = pd.to_datetime(df[col], format='%Y-%m-%d', errors='ignore')

df = df[cols]

df.head()
# Create histogram showing distribution of deaths over time

register_matplotlib_converters()

x = df['date_of_death'].dropna()

bins = pd.cut(x, bins=22)

fig1, ax1 = plt.subplots(1, figsize=(10,8))

(n, b, patches) = ax1.hist(x, bins=22, facecolor="#278fe9", alpha=.7)

ax1.set_title('Distribution of deaths among members of the National Convention', fontsize=15)

ax1.set_xlabel('Date', fontsize=15)

ax1.set_ylabel('Number of deaths', fontsize=15)

ax1.set_xlim(x.min(),x.max())

for spine in ax1.spines:

    ax1.spines[spine].set_visible(False)

patches[0].set_color('#276092')

ax1.annotate('105 deputies died during their mandate between Sep 1792 and Oct 1795', xy=(80, 400),

             xytext=(110, 410), xycoords='figure pixels', clip_on=True,

             arrowprops=dict(facecolor='black', shrink=0.05), fontsize=10)

plt.show()
# create new DataFrame limited to deputies who died during mandate

df2 = df[df.loc[:,('date_of_death')] <= pd.Timestamp(1795,10,26)]

# change datetime to period (Quarters)

df2.loc[:,('quarter')] = pd.PeriodIndex(df2.date_of_death, freq='Q')

# Assign 'unknown' to deputies without explicit group affiliation

df2.loc[:,('group')] = df2.loc[:('group')].fillna('unknown')

# group by 'group' and 'quarter'

df2_grouped = df2.groupby(['group', 'quarter']).size()

# reshape further to create a pivot table

df2_grouped = df2_grouped.reset_index()

df2_grouped.columns = ['group', 'quarter','number_of_deaths']

df2_grouped_pivot = df2_grouped.pivot(index='quarter', columns='group', values='number_of_deaths').fillna(0)

df2_grouped_pivot = df2_grouped_pivot[['Montagne','Gauche','Dantonnistes','Girondins','Modérés',\

                                       'Majorité', 'Plaine', 'Centre droit', 'Droite', 'unknown']]
# Create combined barplot and heatmap

fig2, (ax2,ax3) = plt.subplots(2,1, sharex=False, figsize=(10,10))

fig2.subplots_adjust(top=1.5)



ax2.set_title('Number of deaths during mandate by political group (barchart) and across time (heatmap)',

              fontsize=16)



#Create barplot

sns.barplot(df2_grouped_pivot.columns, df2_grouped_pivot.values.sum(axis=0), color='#e7ba96' ,ci=None,

            ax=ax2)

ax2.tick_params(axis='x', labelbottom=False, bottom=False)

ax2.tick_params(axis='y', labelsize=15)

ax2.set_ylabel('Total number of deaths by political group', fontsize=12)

ax2.set_xlabel('')

for spine in ax2.spines:

    ax2.spines[spine].set_visible(False)



#Create heatmap

cmap = sns.cubehelix_palette(light=1, as_cmap=True)

sns.heatmap(df2_grouped_pivot, vmin=0, vmax=df2_grouped_pivot.values.max(), cmap=cmap, cbar=False,

            linewidth=.5, annot=True, annot_kws={'size': 15}, center=12, ax=ax3)

ax3.tick_params(axis='x', labeltop=True, labelbottom=False, rotation=90, labelsize=12, bottom=False)

ax3.tick_params(axis='y', rotation=0, labelsize=12, left=False)

ax3.set_ylabel('Year/Quarter', fontsize=14)

ax3.set_xlabel('')

plt.show()