import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib as mpl

import matplotlib.pyplot as plt



print('Numpy Version     : ', np.__version__)

print('Pandas Version    : ', pd.__version__)

print('Matplotlib Version: ', mpl.__version__)

print('Seaborn Version   : ', sns.__version__)



#seaborn options

sns.set_style('white')



#pandas options

pd.options.display.max_rows = 100

pd.options.display.max_columns = 100
def missingData(data):

    total = data.isnull().sum().sort_values(ascending = False)

    percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending = False)

    md = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

    md = md[md["Percent"] > 0]

    plt.figure(figsize = (8, 4))

    plt.xticks(rotation='90')

    sns.barplot(md.index, md["Percent"],color="r",alpha=0.8)

    plt.xlabel('Features', fontsize=15)

    plt.ylabel('Percent of missing values', fontsize=15)

    plt.title('Percent missing data by feature', fontsize=15)

    return md



def valueCounts(dataset, features):

    """Display the features value counts """

    for feature in features:

        vc = dataset[feature].value_counts()

        print(vc)
data = pd.read_csv('../input/athlete_events.csv')

regions = pd.read_csv('../input/noc_regions.csv')
data.sample(5)
data.info()
data.describe()
regions.head()
regions.info()
allAthletes = pd.merge(data, regions, on='NOC', how='left')
allAthletes.sample(5)
allAthletes.shape
allAthletes.info()
missingData(allAthletes)
allAthletes['Medal'].fillna('No Medal', inplace=True)

allAthletes.drop('notes', axis=1, inplace=True)
missingData(allAthletes)
allAthletes.describe()
plt.figure(figsize=(20, 10))

sns.countplot(x= 'Age', hue='Sex', data=allAthletes )

plt.xticks(rotation = 90)

plt.legend(loc=1, fontsize='x-large')

plt.title('Distribution of olympic athletes by Age and Sex', fontsize = 20)

plt.show()
plt.figure(figsize=(20, 10))

sns.countplot(allAthletes['Age'], color='b')

plt.xticks(rotation = 90)

plt.title('Distribution of olympic athletes by Age', fontsize = 20)

plt.show()
allAthletes['ID'][allAthletes['Age'] > 60].count()
allAthletes['Age'].max()
allAthletes[allAthletes['Age'] == 97].head()
allAthletesHW = allAthletes[(allAthletes['Height'].notnull()) & (allAthletes['Weight'].notnull())]
allAthletesHW.describe()
plt.figure(figsize=(15, 15))

sns.scatterplot(x="Height", y="Weight", hue='Medal', data=allAthletesHW)

plt.title('Height VS Weight of Olympics Athletes', fontsize=20)

plt.show()
allAthletesHW['Weight'].max()
allAthletesHW[allAthletesHW['Weight'] == 214].head()
allAthletesHW['Height'].max()
allAthletesHW[allAthletesHW['Height'] == 226].head()
athletesMenHW = allAthletesHW[(allAthletesHW.Sex == 'M')]

athletesWomenHW = allAthletesHW[(allAthletesHW.Sex == 'F')]



summerMenHW = athletesMenHW[(athletesMenHW.Season == 'Summer')]

winterMenHW = athletesMenHW[(athletesMenHW.Season == 'Winter')]

summerWomenHW = athletesWomenHW[(athletesWomenHW.Season == 'Summer')]

winterWomenHW = athletesWomenHW[(athletesWomenHW.Season == 'Winter')]



print('Summer Male Athletes shape   : ', summerMenHW.shape)

print('Winter Male Athletes shape   : ', winterMenHW.shape)

print('Summer Female Athletes shape : ', summerWomenHW.shape)

print('Winter Female Athletes shape : ', winterWomenHW.shape)
plt.figure(figsize=(30,18))



#1

plt.subplot(221)

sns.boxplot('Year', 'Weight', data=summerMenHW, palette='Oranges')

plt.title('Variation of Weight for Male Athletes in summer Olympics', fontsize=20)

#2

plt.subplot(222)

sns.boxplot('Year', 'Weight', data=summerWomenHW, palette='Oranges')

plt.title('Variation of Weight for Female Athletes in summer Olympics', fontsize=20)

#3

plt.subplot(223)

sns.boxplot('Year', 'Weight', data=winterMenHW, palette='Blues')

plt.title('Variation of Weight for Male Athletes in winter Olympics', fontsize=20)

#4

plt.subplot(224)

sns.boxplot('Year', 'Weight', data=winterWomenHW, palette='Blues')

plt.title('Variation of Weight for Female Athletes in winter Olympics', fontsize=20)



plt.show()
plt.figure(figsize=(30,18))



#1

plt.subplot(221)

sns.boxplot('Year', 'Height', data=summerMenHW, palette='Oranges')

plt.title('Variation of Height for Male Athletes in summer Olympics', fontsize=20)

#2

plt.subplot(222)

sns.boxplot('Year', 'Height', data=summerWomenHW, palette='Oranges')

plt.title('Variation of Height for Female Athletes in summer Olympics', fontsize=20)

#3

plt.subplot(223)

sns.boxplot('Year', 'Height', data=winterMenHW, palette='Blues')

plt.title('Variation of Height for Male Athletes in winter Olympics', fontsize=20)

#4

plt.subplot(224)

sns.boxplot('Year', 'Height', data=winterWomenHW, palette='Blues')

plt.title('Variation of Height for Female Athletes in winter Olympics', fontsize=20)



plt.show()
athletesMen = allAthletes[(allAthletes.Sex == 'M')]

athletesWomen = allAthletes[(allAthletes.Sex == 'F')]



print('Male Athletes shape   : ', athletesMen.shape)

print('Female Athletes shape : ', athletesWomen.shape)
athletesMen.describe()
athletesWomen.describe()
summerMen = athletesMen[(athletesMen.Season == 'Summer')]

winterMen = athletesMen[(athletesMen.Season == 'Winter')]

summerWomen = athletesWomen[(athletesWomen.Season == 'Summer')]

winterWomen = athletesWomen[(athletesWomen.Season == 'Winter')]



print('Summer Male Athletes shape   : ', summerMen.shape)

print('Winter Male Athletes shape   : ', winterMen.shape)

print('Summer Female Athletes shape : ', summerWomen.shape)

print('Winter Female Athletes shape : ', winterWomen.shape)
summerTicks = list(summerMen['Year'].unique())

summerTicks.sort()

winterTicks = list(winterMen['Year'].unique())

winterTicks.sort()
plt.figure(figsize=(30,18))



plt.subplot(221)

partSummerMen = summerMen.groupby('Year')['Sex'].value_counts()

partSummerMen.loc[:,'M'].plot(linewidth=4, color='b')

plt.xticks(summerTicks, rotation=90)

plt.title('Variation of Male Athletes in summer Olympics', fontsize=20)



plt.subplot(222)

partSummerWomen = summerWomen.groupby('Year')['Sex'].value_counts()

partSummerWomen.loc[:,'F'].plot(linewidth=4, color='r')

plt.xticks(summerTicks, rotation=90)

plt.title('Variation of Female Athletes in summer Olympics', fontsize=20)



plt.subplot(223)

partWinterMen = winterMen.groupby('Year')['Sex'].value_counts()

partWinterMen.loc[:,'M'].plot(linewidth=4, color='b')

plt.xticks(winterTicks, rotation=90)

plt.title('Variation of Male Athletes in winter Olympics', fontsize=20)



plt.subplot(224)

partWinterWomen = winterWomen.groupby('Year')['Sex'].value_counts()

partWinterWomen.loc[:,'F'].plot(linewidth=4, color='r')

plt.xticks(winterTicks, rotation=90)

plt.title('Variation of Female Athletes in winter Olympics', fontsize=20)



plt.show()
#crate a dataset with gold medals

goldMedals = allAthletes[(allAthletes.Medal == 'Gold')]
goldMedals.sample(5)
goldMedals.info()
missingData(goldMedals)
allAthletesAge = allAthletes[(allAthletes['Age'].notnull())]
missingData(allAthletesAge)
goldMedals.describe()
plt.figure(figsize=(20, 10))

sns.countplot(x= 'Age', hue='Sex', data=goldMedals )

plt.xticks(rotation = 90)

plt.legend(loc=1, fontsize='x-large')

plt.title('Distribution of Gold Medals by Age and Sex', fontsize = 20)

plt.show()
goldMedals['ID'][goldMedals['Age']<18].count()
minorsAthletes = goldMedals[goldMedals['Age']<18]
plt.figure(figsize=(20, 10))

sns.countplot(x= 'Sport', hue='Sex', data=minorsAthletes)

plt.xticks(rotation = 90)

plt.title('Minors Gold Athletes by Sport', fontsize=20)

plt.legend(loc=1, fontsize='x-large')

plt.show()
plt.figure(figsize=(20, 10))

sns.countplot(x= 'Event', hue='Sex', data=minorsAthletes)

plt.xticks(rotation=90)

plt.title('Minors Gold Athletes by Event', fontsize=20)

plt.legend(loc=1, fontsize='x-large')

plt.show()
goldMedals['ID'][goldMedals['Age'] == 13].count()
youngestAthletes = goldMedals[goldMedals['Age'] == 13]
youngestAthletes
plt.figure(figsize=(20, 10))

ticks = [0, 1, 2, 3]

sns.countplot(x='Sport', hue='Sex', data=youngestAthletes)

plt.yticks(ticks)

plt.title('Gold Athletes thirteen', fontsize=20)

plt.legend(loc=1, fontsize='x-large')

plt.show()
goldMedals['ID'][goldMedals['Age'] > 50].count()
oldAthletes = goldMedals[goldMedals['Age'] > 50]
plt.figure(figsize=(20, 10))

ticks = [0, 5, 10, 15, 20]

plt.yticks(ticks)

sns.countplot(x = 'Sport', hue='Sex', data=oldAthletes)

plt.title('Gold Athletes over 50 by sport', fontsize=20)

plt.legend(loc=1, fontsize='x-large')

plt.show()
plt.figure(figsize=(20, 10))

sns.countplot(x='Year', hue='Season', data=goldMedals)

plt.title('Gold medals per edition of the Games', fontsize = 20)

plt.legend(loc=2, fontsize='x-large')

plt.show()
goldMedals.region.value_counts().reset_index(name='Medal').head(10)
totalGoldMedals = goldMedals.region.value_counts().reset_index(name='Medal').head(5)



sns.catplot(x="index", y="Medal", data=totalGoldMedals,

                height=6, kind="bar", palette='afmhot')

plt.xlabel("Countries")

plt.ylabel("Number of Medals")

plt.title('Top 5 Countries', fontsize = 20)

plt.show()
goldMedals.sample(5)
goldMedals.info()
goldMedalsHW = goldMedals[(goldMedals['Height'].notnull()) & (goldMedals['Weight'].notnull())]
markers = {"Summer": "d", "Winter": "h"}



plt.figure(figsize=(15, 15))

sns.scatterplot(x="Height", y="Weight", hue='Sex', style='Season', data=goldMedalsHW, markers=markers)

plt.title('Height VS Weight of Gold Medalists', fontsize=20)

plt.legend(loc=1, fontsize='x-large')

plt.show()

goldOver140kg = goldMedalsHW.loc[goldMedalsHW['Weight'] > 140]
valueCounts(goldOver140kg, ['Sport'])
goldOver140kg
over200cm = goldMedalsHW.loc[goldMedalsHW['Height'] > 200]
valueCounts(over200cm, ['Sport'])
itAthletes = allAthletes[(allAthletes.region == 'Italy')].sort_values('Year')
itAthletes.info()
plt.figure(figsize=(20, 10))

sns.countplot(x='Year', hue='Season',data=itAthletes)

plt.title('Italian athletes per edition of the Games by Season', fontsize = 20)

plt.legend(loc=2, fontsize='x-large')

plt.show()
plt.figure(figsize=(20, 10))

sns.countplot(x='Year',hue='Sex', data=itAthletes)

plt.title('Italian athletes per edition of the Games by Sex', fontsize = 20)

plt.legend(loc=1, fontsize='x-large')

plt.show()
itMen = itAthletes[(itAthletes.Sex == 'M')]

itWomen = itAthletes[(itAthletes.Sex == 'F')]
markers = {"Summer": "d", "Winter": "h"}



plt.figure(figsize=(15, 15))

sns.scatterplot(x="Height", y="Weight", hue='Medal', style='Season', data=itAthletes.sort_values('Year'), markers=markers)

plt.title('Height VS Weight of Olympics Italian Athletes', fontsize=20)

plt.legend(loc=1, fontsize='x-large')

plt.show()
goldMedalsITA = itAthletes[(itAthletes.Medal == 'Gold')]
plt.figure(figsize=(20, 10))

sns.countplot(x='Age', hue='Sex', data=goldMedalsITA)

plt.title('Distribution of Italian Gold Medals by Age', fontsize = 20)

plt.legend(loc=1, fontsize='x-large')

plt.show()
plt.figure(figsize=(20, 10))

sns.countplot(goldMedalsITA['Age'], color='b')

plt.title('Distribution of Italian Gold Medals by Age', fontsize = 20)

plt.show()
plt.figure(figsize=(20, 10))

sns.countplot(x='Year',hue='Sex', data=goldMedalsITA)

plt.title('Italian Gold medals per edition of the Games', fontsize = 20)

plt.legend(loc=1, fontsize='x-large')

plt.show()
plt.figure(figsize=(20, 10))

sns.countplot(x='Year', hue='Sex', data=goldMedalsITA)

plt.title('Distribution of Italian Gold Medals by Year', fontsize = 20)

plt.legend(loc=1, fontsize='x-large')

plt.show()
goldMedalsITA.Event.value_counts().reset_index(name='Medal').head(10)
itGoldMen = goldMedalsITA[(goldMedalsITA.Sex == 'M')]

itGoldWomen = goldMedalsITA[(goldMedalsITA.Sex == 'F')]
plt.figure(figsize=(20, 10))

sns.countplot(x='Year', data=itGoldMen, color='b')

plt.title('Male Italian Gold medals per edition of the Games', fontsize = 20)

plt.show()
plt.figure(figsize=(20, 10))

sns.countplot(x='Year', data=itGoldWomen, color='r')

plt.title('Female Italian Gold medals per edition of the Games', fontsize = 20)

plt.show()
plt.figure(figsize=(30,20))



#1

plt.subplot(221)

sns.countplot(x='Year', hue='Season', data=itMen)

plt.xticks(rotation=90)

plt.title('Italian Men per edition of the Games', fontsize = 20)

plt.legend(loc=2, fontsize='large')



#2

plt.subplot(222)

sns.countplot(x='Year', hue='Season', data=itWomen)

plt.xticks(rotation=90)

plt.title('Italian Women per edition of the Games', fontsize = 20)

plt.legend(loc=2, fontsize='large')

#3

plt.subplot(223)

sns.countplot(x='Year', hue='Season', data=itGoldMen)

plt.xticks(rotation=90)

plt.title('Male Italian Gold medals per edition of the Games', fontsize = 20)

plt.legend(loc=2, fontsize='large')

#4

plt.subplot(224)

sns.countplot(x='Year', hue='Season', data=itGoldWomen)

plt.xticks(rotation=90)

plt.title('Female Italian Gold medals per edition of the Games', fontsize = 20)

plt.legend(loc=2, fontsize='large')



plt.show()
markers = {"Summer": "d", "Winter": "h"}



plt.figure(figsize=(15, 15))

sns.scatterplot(x="Height", y="Weight", hue='Sex',style='Season', data=goldMedalsITA)

plt.title('Height VS Weight of Italian Gold Medals', fontsize=20)

plt.show()