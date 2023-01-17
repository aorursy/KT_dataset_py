# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Read our data for Global
df = pd.read_csv('../input/globalterrorismdb_0718dist.csv', encoding='windows-1252')
list(df.columns)
df.head()
turkeyDf = df[(df.country_txt == 'Turkey')]  # Data for Turkey
# 30 countries that the most terrorrist attacks occured
most30Global = df.country_txt.value_counts()[:31]
plt.figure(figsize = (15,10))
ax = sns.barplot(x=most30Global,y=most30Global.index, palette="rocket")
plt.title('30 Countries Where Most of the Terrorist Attacks were Shown')
plt.xlabel('Numbers of Terrorirst Attacks (1970 - 2017)')
plt.xticks(np.arange(0,26500,2000))
plt.show()
df.region_txt.value_counts()
dfRegion = df.groupby('region_txt').sum()

dfRegionAndPeopleKilled = sorted(list(zip(dfRegion['nkill'].values, dfRegion['nkill'].index)), reverse=True)
peopleWereKilledList, regionList = zip(*dfRegionAndPeopleKilled)
peopleWereKilledList, regionList = list(peopleWereKilledList), list(regionList)

plt.figure(figsize=(15,10))
sns.barplot(x=regionList, y=peopleWereKilledList, palette= sns.color_palette("YlOrRd_r", 10))
plt.xticks(rotation=45, ha='right')
plt.title("Numbers of people killed")
plt.show()

# Percentage of Attacks By Regions
labels = dfRegion.index
colors=['#17A50E', '#D905FA', '#D0E02F', '#2FD0E0', '#FF0000', '#00FFE8','#CAC8C2', '#6AF394', '#C08CEC', '#FAA6E5', '#37528A', '#DF650C']
explode=[0,0,0,0,0,0,0,0,0,0,0,0]
sizes = dfRegion['nkill'].values

plt.figure(figsize=(15,15))
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', pctdistance=0.5)
plt.title('Terrorist Attacks by Regions')
plt.show()
# Terrorist Incidents by Years
dfYear = list(zip(df.iyear.value_counts().index, df.iyear.value_counts().values))
dfYear = sorted(dfYear)
terrorByYear = []
terrorByCounts = []
for i in dfYear:
    terrorByYear.append(i[0])
    terrorByCounts.append(i[1])
    
plt.figure(figsize=(20,5))
sns.barplot(x=terrorByYear, y=terrorByCounts, palette=sns.color_palette("hls", len(terrorByYear)))
plt.xticks(rotation=90)
plt.xlabel('Year')
plt.ylabel('Numbers of Attack')
plt.show()
totalKilled = df[df['iyear'] <= 2007].groupby('gname').sum()['nkill'].values.sum()
totalKilledAfter2007 = df[df['iyear'] > 2007].groupby('gname').sum()['nkill'].values.sum()
print("Total people killed by terrorists between 1970 and 2007:",int(totalKilled))
print("Total people killed by terrorists between 2008 and 2017:",int(totalKilledAfter2007))

df[df['iyear'] > 2007].groupby('gname').sum()[['nkill']].sort_values(['nkill'],ascending=False)
byYear = df.groupby("iyear").sum()
plt.figure(figsize=(20,5))
byYear.nkill.plot(color="#4F098E", label = "Number of People Killed", linewidth = 4, alpha = 0.8, grid = True)
plt.legend(loc='upper left')
plt.xticks(np.arange(byYear.index.min(),2018,5))
plt.yticks(np.arange(0,50000,5000))
plt.xlabel('Year')
plt.ylabel('People Killed')
plt.title('Number of People Killed')
plt.show()
df.head()
regionAndYear = pd.crosstab(df.iyear, df.region_txt)
regionAndYear.plot(color=sns.color_palette("bright", 12),grid=True, linewidth = 2)
figure=plt.gcf()
figure.set_size_inches(20,6)
plt.xlabel("Year")
plt.ylabel("Numbers of Terrorist Attacks")
plt.title("Terrorist Attacks by Year and Region")
plt.show()
df['attacktype1_txt'].value_counts()
plt.figure(figsize=(20,8))
sns.set(style="darkgrid", context="talk")
sns.barplot(x=df['attacktype1_txt'].value_counts().index, y=df['attacktype1_txt'].value_counts().values, palette="GnBu_r")
plt.xlabel("Methods of Attacks")
plt.ylabel("Numbers of Attacks")
plt.xticks(rotation=45, ha="right")
plt.title("Attacks Type")
plt.show()

turkeyDf.head()
yearOfAttacksTurkey = turkeyDf['iyear'].value_counts()
listOfAttacksYear = sorted(list(zip(yearOfAttacksTurkey.index, yearOfAttacksTurkey.values)))
attacksYear, attacksCount = zip(*listOfAttacksYear)
attacksYear, attacksCount = list(attacksYear), list(attacksCount)

sns.set(style="white", context="talk")
f, ax1 = plt.subplots(figsize=(20,10))
sns.pointplot(x=attacksYear, y=attacksCount, color="orange")
plt.xlabel("Year")
plt.xticks(rotation=90)
plt.ylabel("Numbers of Terrorist Attacks")
plt.grid()


mostActiveGroupsTen = turkeyDf[turkeyDf['gname'].isin(turkeyDf['gname'].value_counts()[:11].index)]
sns.set(style="whitegrid", context="talk")
pd.crosstab(mostActiveGroupsTen.iyear,mostActiveGroupsTen.gname).plot(color=sns.color_palette('Paired',12))
plt.xlabel("Year")
fig=plt.gcf()
fig.set_size_inches(20,6)
plt.show()
terrorGroupsTurkey = turkeyDf.gname.value_counts()
terrorGroupsList15 = sorted(list(zip(terrorGroupsTurkey.values[:16], terrorGroupsTurkey.index[:16])), reverse=True)
attacksInTurkey15, terroristsInTurkey15 = zip(*terrorGroupsList15)
attacksInTurkey15, terroristsInTurkey15 = list(attacksInTurkey15), list(terroristsInTurkey15)

plt.figure(figsize=(20,6))
sns.barplot(x=terroristsInTurkey15, y=attacksInTurkey15, palette = sns.cubehelix_palette(len(terroristsInTurkey15)))
plt.xticks(rotation=70, ha="right")
plt.xlabel("Terrorist Organisations")
plt.ylabel("Attacks")
plt.show()
citiesAttacked = turkeyDf.provstate.value_counts()
citiesAttackedList = sorted(list(zip(citiesAttacked.values[:31], citiesAttacked.index[:31])), reverse=True)
citiesAttacked30, citiesCount30 = zip(*citiesAttackedList)
citiesAttacked30, citiesCount30 = list(citiesAttacked30), list(citiesCount30)

plt.figure(figsize=(20,10))
sns.barplot(x=citiesAttacked30, y=citiesCount30)
plt.xticks(np.arange(0,1201,100))
plt.xlabel("Number of Attacks")
plt.show()
attackType = turkeyDf.attacktype1_txt.value_counts()
attackTypeList = sorted(list(zip(attackType.values, attackType.index)), reverse=True)
attackCount, attackMethod = zip(*attackTypeList)
attackCount, attackMethod = list(attackCount), list(attackMethod)

plt.figure(figsize=(20,8))
sns.barplot(x=attackMethod, y=attackCount, palette = sns.color_palette("YlOrBr_r", 9))
plt.xticks(rotation=35, ha="right")
plt.xlabel("Attack Types of Terrorists")
plt.title("Attack Types")
plt.ylabel("Attacks")
plt.show()
targetType = turkeyDf.targtype1_txt.value_counts()
targetTypeList = sorted(list(zip(targetType.values, targetType.index)), reverse=True)
targetCount, target = zip(*targetTypeList)
targetCount, target = list(targetCount), list(target)

plt.figure(figsize=(20,8))
sns.barplot(x=target, y=targetCount, palette = sns.color_palette("YlGn_r", 21))
plt.xticks(rotation=35, ha="right")
plt.xlabel("Targets of Terrorists")
plt.ylabel("Attacks")
plt.show()
terrorists=turkeyDf['gname'].value_counts()[:5].to_frame()
terrorists.columns=['gname']
kill=turkeyDf.groupby('gname')['nkill'].sum().to_frame()
terrorists.merge(kill,left_index=True,right_index=True,how='left').plot.bar(width=0.9)
fig=plt.gcf()
fig.set_size_inches(20,6)
plt.show()