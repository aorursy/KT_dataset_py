import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')

events=pd.read_csv('/kaggle/input/120-years-of-olympic-history-athletes-and-results/athlete_events.csv')
events.head()
region=pd.read_csv('/kaggle/input/120-years-of-olympic-history-athletes-and-results/noc_regions.csv')
region.head()
merged = pd.merge(events, region, on='NOC', how='left')
merged.head()
GOLDMEDALS = merged[(merged.Medal == 'Gold')]

GOLDMEDALS.head()
GOLDMEDALS.isnull().sum()
plt.figure(figsize=(20, 10))

plt.tight_layout()

sns.countplot(GOLDMEDALS['Age'])

plt.title('Distribution of Gold Medals')
GOLDMEDALS['ID'][GOLDMEDALS['Age'] > 50].count()
GOLDMEDALS['ID'][GOLDMEDALS['Age'] > 60].count()
DISCIPLINES = GOLDMEDALS['Sport'][GOLDMEDALS['Age'] > 50]
plt.figure(figsize=(20, 10))

plt.tight_layout()

sns.countplot(DISCIPLINES)

plt.title('Gold Medals for Athletes Over 50')
GOLDMEDALS.region.value_counts().reset_index(name='Medal').head(5)
TOTALGOLDMEDALS = GOLDMEDALS.region.value_counts().reset_index(name='Medal').head(5)

g = sns.catplot(x="index", y="Medal", data=TOTALGOLDMEDALS,

                height=6, kind="bar", palette="muted")

g.despine(left=True)

g.set_xlabels("Top 5 countries")

g.set_ylabels("Number of Medals")

plt.title('Medals per Country')
GOLDMEDALSUSA = GOLDMEDALS.loc[GOLDMEDALS['NOC'] == 'USA']
GOLDMEDALSUSA.Event.value_counts().reset_index(name='Medal').head(20)
BASKETBALLGOLDUSA = GOLDMEDALSUSA.loc[(GOLDMEDALSUSA['Sport'] == 'Basketball') & (GOLDMEDALSUSA['Sex'] == 'M')].sort_values(['Year'])
BASKETBALLGOLDUSA.head()
NOTNullMedals = GOLDMEDALS[(GOLDMEDALS['Height'].notnull()) & (GOLDMEDALS['Weight'].notnull())]
NOTNullMedals.head()
plt.figure(figsize=(12, 10))

ax = sns.scatterplot(x="Height", y="Weight", data=NOTNullMedals)

plt.title('Height vs Weight of Olympic Medalists')
NOTNullMedals.loc[NOTNullMedals['Weight'] > 160]