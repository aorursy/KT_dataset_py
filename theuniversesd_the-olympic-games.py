import numpy as np

import pandas as pd

import seaborn as sns

from matplotlib import pyplot as plt



import os

print(os.listdir("../input/120-years-of-olympic-history-athletes-and-results"))
data = pd.read_csv('../input/120-years-of-olympic-history-athletes-and-results/athlete_events.csv')

regions = pd.read_csv('../input/120-years-of-olympic-history-athletes-and-results/noc_regions.csv')
data.head()
data.describe()
data.info()
regions.head()
merged = pd.merge(data, regions, on='NOC', how='left')
merged.head()
goldMedals = merged[(merged.Medal == 'Gold')]

goldMedals.head()
goldMedals.isnull().any()
goldMedals = goldMedals[np.isfinite(goldMedals['Age'])]
plt.figure(figsize=(20,10))

plt.tight_layout()

sns.countplot(goldMedals['Age'])

plt.title('Distribution of Gold Medals')
goldMedals['ID'][goldMedals['Age']> 50].count()
masterDisciplines = goldMedals['Sport'][goldMedals['Age'] > 50]
plt.figure(figsize=(20, 10))

plt.tight_layout()

sns.countplot(masterDisciplines)

plt.title('Gold Medals for Athletes Over 50')
goldMedals['ID'][goldMedals['Age']< 15].count()
masterDisciplines1 = goldMedals['Sport'][goldMedals['Age'] < 15]
plt.figure(figsize=(20, 10))

plt.tight_layout()

sns.countplot(masterDisciplines1)

plt.title('Gold Medals for Athletes below 15')
womenInOlympics = merged[(merged.Sex == 'F') & (merged.Season == 'Summer')]
womenInOlympics.head(10)
sns.set(style="darkgrid")

plt.figure(figsize=(20, 10))

sns.countplot(x='Year', data=womenInOlympics)

plt.title('Women medals per edition of the Games')
womenInOlympics.loc[womenInOlympics['Year'] == 1900].head(10)
womenInOlympics['ID'].loc[womenInOlympics['Year'] == 1900].count()
goldMedals.region.value_counts().reset_index(name='Medals').head(30)
totalGoldMedals = goldMedals.region.value_counts().reset_index(name='Medal').head()

g = sns.catplot(x="index", y="Medal", data=totalGoldMedals, height=6, kind="bar", palette="muted")

g.despine(left=True)

g.set_xlabels('Top 5 countries')

g.set_ylabels('Number of Medals')

plt.title('Medals per Country')
goldMedalsIndia = goldMedals.loc[goldMedals['NOC'] == 'India']
goldMedalsIndia.Event.value_counts().reset_index(name='Medal').head()
goldMedalsUSA = goldMedals.loc[goldMedals['NOC'] == 'USA']
goldMedalsUSA.Event.value_counts().reset_index(name='Medal').head(20)
basketballGoldUSA = goldMedalsUSA.loc[(goldMedalsUSA['Sport'] == 'Basketball') & (goldMedalsUSA['Sex'] == 'M')].sort_values(['Year'])
basketballGoldUSA.head(15)
groupedBasketUSA = basketballGoldUSA.groupby(['Year']).first()

groupedBasketUSA
groupedBasketUSA['ID'].count()
goldMedals.head()
goldMedals.info()
notNullMedals = goldMedals[(goldMedals['Height'].notnull()) & (goldMedals['Weight'].notnull())]
notNullMedals.head()
notNullMedals.info()
plt.figure(figsize=(12, 10))

ax = sns.scatterplot(x="Height", y="Weight", data=notNullMedals)

plt.title('Height vs Weight of Olympic Medalists')
notNullMedals.loc[notNullMedals['Weight'] > 160]
MenOverTime = merged[(merged.Sex == 'M') & (merged.Season == 'Summer')]

WomenOverTime = merged[(merged.Sex == 'F') & (merged.Season == 'Summer')]
MenOverTime.head()
part = MenOverTime.groupby('Year')['Sex'].value_counts()

plt.figure(figsize=(20, 10))

part.loc[:,'M'].plot()

plt.title('Variation of Male Athletes over time')
part = WomenOverTime.groupby('Year')['Sex'].value_counts()

plt.figure(figsize=(20, 10))

part.loc[:,'F'].plot()

plt.title('Variation of Female Athletes over time')
plt.figure(figsize=(20, 10))

sns.boxplot('Year', 'Age', data=MenOverTime)

plt.title('Variation of Age for Male Athletes over time')
MenOverTime.loc[MenOverTime['Age'] > 80].head(10)
plt.figure(figsize=(20, 10))

sns.boxplot('Year', 'Age', data=WomenOverTime)

plt.title('Variation of Age for Female Athletes over time')
WomenOverTime.loc[WomenOverTime['Year'] == 1904]
plt.figure(figsize=(20, 10))

sns.pointplot('Year', 'Weight', data=MenOverTime)

plt.title('Variation of Weight for Male Athletes over time')
plt.figure(figsize=(20, 10))

sns.pointplot('Year', 'Weight', data=WomenOverTime)

plt.title('Variation of Weight for Female Athletes over time')
womenInOlympics.loc[womenInOlympics['Year'] < 1924].head(20)
plt.figure(figsize=(20, 10))

sns.pointplot('Year', 'Height', data=MenOverTime, palette='Set2')

plt.title('Variation of Height for Male Athletes over time')
plt.figure(figsize=(20, 10))

sns.pointplot('Year', 'Height', data=WomenOverTime, palette='Set2')

plt.title('Variation of Height for Female Athletes over time')
WomenOverTime.loc[(WomenOverTime['Year'] > 1924) & (WomenOverTime['Year'] < 1952)].head(10)
itMenOverTime = MenOverTime.loc[MenOverTime['region'] == 'India']
itMenOverTime.head(5)
sns.set(style="darkgrid")

plt.figure(figsize=(20, 10))

sns.countplot(x='Year', data=itMenOverTime, palette='Set2')

plt.title('Variation of Age for Indian Male Athletes over time')
itWomenOverTime = WomenOverTime.loc[WomenOverTime['region'] == 'India']
itWomenOverTime.head()
sns.set(style="darkgrid")

plt.figure(figsize=(20, 10))

sns.countplot(x='Year', data=itWomenOverTime, palette='Set2')

plt.title('Variation of Age for Indian Female Athletes over time')
wlWomenOverTime = WomenOverTime.loc[WomenOverTime['Sport'] == 'Weightlifting']
plt.figure(figsize=(20, 10))

sns.pointplot('Year', 'Height', data=wlWomenOverTime)

plt.title('Height over year for Female Lifters')
wlWomenOverTime['Weight'].loc[wlWomenOverTime['Year'] < 2000].isnull().all()