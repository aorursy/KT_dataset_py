import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
eve=pd.read_csv("/kaggle/input/120-years-of-olympic-history-athletes-and-results/athlete_events.csv")
eve.head()
eve.tail()
eve.shape
eve.isnull().sum()
eve.info()
eve = eve.drop_duplicates(keep='first')
eve.shape
eve['Year'].max()
eve['Year'].min()
eve.describe()
sns.pairplot(eve)
# distribution plot with 10 bins
tpl = tuple(eve['Height'])
sns.distplot(tpl, bins=10)
# distribution plot with 10 bins
tpl = tuple(eve['Weight'])
sns.distplot(tpl, bins=10)
# distribution plot with 10 bins
tpl = tuple(eve['Age'])
sns.distplot(tpl, bins=10)
# distribution plot with 10 bins
tpl = tuple(eve['Year'])
sns.distplot(tpl, bins=10)
eve['Age'].unique()
eve['Year'].unique()
eve['Height'].unique()
eve['Weight'].unique()
eve['Team'].unique()
eve['NOC'].unique()
eve['Games'].unique()
eve['Season'].unique()
eve['City'].unique()
eve['Sport'].unique()
eve['Sex'].unique()
eve['Medal'].unique()
FemaleAthletes = eve[(eve.Sex == 'F') & (eve.Height < 150) & (eve.Age<35)]
FemaleAthletes.shape
sns.set(style="ticks")
plt.figure(figsize=(10,5))
sns.countplot(x='Age', data=FemaleAthletes)
plt.title('FemaleAthletes with Age(<35) and Height(<150)')
AthletesInOlympics = eve[(eve.Year == 2012) & (eve.Age>35)]
AthletesInOlympics.shape
plt.figure(figsize=(10, 5))
sns.boxplot( 'Year','Age', data=AthletesInOlympics, palette='Dark2')
plt.title('Variation of Athletes in the Year 2012 with Age(>35)')
maleAthlete = eve[(eve.Age>30) & (eve.Age<50) & (eve.Sex=='M')&(eve.City=='London')]
maleAthlete.shape
sns.set(style="darkgrid")
sns.countplot(y="Age",hue="City",data=maleAthlete,palette="CMRmap",)
noOfAthletes=eve[(eve.City=='Paris')&(eve.Medal=='Gold')]
noOfAthletes.shape
sns.boxplot(x='Year',y='Age', data=noOfAthletes, palette='cool')
gamesInSeoul=eve[(eve.Sport=='Judo')&(eve.City=='Seoul')&(eve.Season=='Summer')]
gamesInSeoul.shape
plt.figure(figsize=(20,5))
sns.barplot('Age', 'Weight', data=gamesInSeoul)
plt.title('Judo Sport played during Summer in Seoul')
sportOfWrestling = eve[(eve.Sport == 'Wrestling') & (eve.Year<1950)]
sportOfWrestling.shape
sportOfWrestling = eve[(eve.Sport == 'Wrestling') & (eve.Year<1950)]
plt.figure(figsize=(20, 10))
sns.pointplot('Year', 'Height', data=sportOfWrestling, palette='cubehelix')
plt.title('Heights of Athletes in Wrestling before Year 1950')
teamMedal=eve[(eve.Medal=='Gold')&(eve.Sex=='M')&(eve.Sport=='Ice Hockey')]
teamMedal.shape
plt.figure(figsize=(20, 10))
sns.pointplot('Year', 'Weight', data=teamMedal)
plt.title('Weights of Men who won Gold Medal in Sport Ice Hockey')
teamFinland=eve[(eve.Team=='Finland')&(eve.Medal!='Gold')&(eve.City=='Beijing')]
teamFinland.shape
sns.lmplot(x="Age", y="Year", hue="Medal", col="Medal",data=teamFinland, height=6, aspect=.7, x_jitter=.1)
saltlakeSport=eve[['Sport']][eve.City=='Salt Lake City']
saltlakeSport.shape
gymnasticAthletes = eve[['Year']][(eve.Year>1990)&(eve.Year<2000)&(eve.Sport=='Gymnastics')]
gymnasticAthletes.shape
tpl = tuple(gymnasticAthletes['Year'])
sns.distplot(tpl, bins=1)
# do not display kernal density estimate (kde) line. display only histogram
sns.distplot(tpl, kde=False, bins=range(3,10,10))
part = eve.groupby('Year')['Sex'].value_counts()
plt.figure(figsize=(20, 10))
part.loc[:,'M'].plot()
plt.title('Variation of Male Athletes over time')
sns.pairplot(eve, hue='Season', palette='summer', markers= ['d','s'],vars=['Year','Age'])
sns.pairplot(eve, palette='hot', hue='Sex',vars=['Year', 'Age'],markers= ['o','d'])
eve['Event'].unique()