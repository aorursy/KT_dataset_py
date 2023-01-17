import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)

%matplotlib inline
df = pd.read_csv('../input/database.csv', low_memory=False)
df.columns = df.columns.map(lambda x: x.replace(' ', '_'))
df.dtypes.tail(8)
df = df[df.Perpetrator_Age.str.contains(' ') == False]
df['Perpetrator_Age'] = pd.to_numeric(df['Perpetrator_Age'], errors='coerce')
df = df[df['Crime_Solved'] == "Yes"]

df = df[df['Relationship'] != 'Unknown']

df = df[df['Crime_Type'] == 'Murder or Manslaughter']

df = df[df['Victim_Sex'] != 'Unknown']
df = df[df['Victim_Sex'] != 'Unknown']

df = df[df['Perpetrator_Age'] >= 2]

df = df[df['Victim_Count'] < 1]

df = df[df['Perpetrator_Count'] == 0]
##Take out mass killings, such as OKC Bombing of 1995

df = df.sort_values('Victim_Count').drop_duplicates(subset=['Agency_Code',

                                                        'Agency_Name', 'Agency_Type',

                                                        'City', 'State', 'Year', 'Month',

                                                        'Perpetrator_Race', 'Perpetrator_Age',

                                                        'Relationship'], keep='last')
plt.figure(figsize=(12,15),facecolor='#efefef')

sns.set()

# ax.set_ticklabels(['0%', '20%', '75%', '100%'])

ax = sns.heatmap(pd.crosstab(df.Relationship,df.Perpetrator_Sex).apply(lambda r: r/r.sum(), axis=1), annot=True, fmt=".0%", linewidths=.5,cmap='Blues')

ax.set_title('Victim Relationship to Perpetrator Sex')

cbar = ax.collections[0].colorbar

cbar.set_ticks([0, .25, .50, .75, 1])

cbar.set_ticklabels(['0%', '25%', '50%',  '75%', '100%'])
plt.figure(figsize=(12,12),facecolor='#efefef')

sns.set()

# ax.set_ticklabels(['0%', '20%', '75%', '100%'])

ax = sns.heatmap(pd.crosstab(df.Relationship,df.Weapon).apply(lambda r: r/r.sum(), axis=1), annot=True, fmt=".0%", linewidths=.5,cmap='Blues')

ax.set_title('Victim Relationship to Perpetrator vs Weapon Used')

cbar = ax.collections[0].colorbar

cbar.set_ticks([0, .25, .50, .75, 1])

cbar.set_ticklabels(['0%', '25%', '50%',  '75%', '100%'])
plt.figure(figsize=(12,12),facecolor='#efefef')

sns.set()

# ax.set_ticklabels(['0%', '20%', '75%', '100%'])

ax = sns.heatmap(pd.crosstab(df.Relationship,df.Month).apply(lambda r: r/r.sum(), axis=1), annot=True, fmt=".0%", linewidths=.5,cmap='Blues')

ax.set_title('Victim Relationship to Perpetrator vs Month')

cbar = ax.collections[0].colorbar

cbar.set_ticks([0, .25, .50, .75, 1])

cbar.set_ticklabels(['0%', '25%', '50%',  '75%', '100%'])
plt.figure(figsize=(12,12),facecolor='#efefef')

sns.set()

# ax.set_ticklabels(['0%', '20%', '75%', '100%'])

ax = sns.heatmap(pd.crosstab(df.Victim_Sex,df.Weapon).apply(lambda r: r/r.sum(), axis=1), annot=True, fmt=".0%", linewidths=.5,cmap='Blues')

ax.set_title('Perpetrator Gender vs Weapon Use')

cbar = ax.collections[0].colorbar

cbar.set_ticks([0, .25, .50, .75, 1])

cbar.set_ticklabels(['0%', '25%', '50%',  '75%', '100%'])
plt.figure(figsize=(12,12),facecolor='#efefef')

sns.set()

# ax.set_ticklabels(['0%', '20%', '75%', '100%'])

ax = sns.heatmap(pd.crosstab(df.Weapon,df.Perpetrator_Sex).apply(lambda r: r/r.sum(), axis=1), annot=True, fmt=".0%", linewidths=.5,cmap='Blues')

ax.set_title('Weapon Use vs Gender')

cbar = ax.collections[0].colorbar

cbar.set_ticks([0, .25, .50, .75, 1])

cbar.set_ticklabels(['0%', '25%', '50%',  '75%', '100%'])