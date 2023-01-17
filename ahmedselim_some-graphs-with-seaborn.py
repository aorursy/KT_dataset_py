# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from collections import Counter

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
PercentOver25CompletedHighSchool  = pd.read_csv('../input/PercentOver25CompletedHighSchool.csv', encoding = 'windows-1254');

MedianHouseholdIncome2015  = pd.read_csv('../input/MedianHouseholdIncome2015.csv', encoding = 'windows-1254');

ShareRaceByCity  = pd.read_csv('../input/ShareRaceByCity.csv', encoding = 'windows-1254');

PoliceKillingsUS  = pd.read_csv('../input/PoliceKillingsUS.csv', encoding = 'windows-1254');

PercentagePeopleBelowPovertyLevel  = pd.read_csv('../input/PercentagePeopleBelowPovertyLevel.csv', encoding = 'windows-1254');
PercentagePeopleBelowPovertyLevel.head()
PercentagePeopleBelowPovertyLevel.info()
PercentagePeopleBelowPovertyLevel.poverty_rate.replace(['-'], 0.0, inplace = True)

PercentagePeopleBelowPovertyLevel.poverty_rate = PercentagePeopleBelowPovertyLevel.poverty_rate.astype('float')

area_list = list(PercentagePeopleBelowPovertyLevel['Geographic Area'].unique())

area_poverty = []

for i in area_list:

    x = PercentagePeopleBelowPovertyLevel[PercentagePeopleBelowPovertyLevel['Geographic Area']==i]

    area_poverty_rate = sum(x.poverty_rate) / len(x.poverty_rate)

    area_poverty.append(area_poverty_rate)

data = pd.DataFrame({'area_list': area_list, 'area_poverty_ratio': area_poverty})

new_index = (data.area_poverty_ratio.sort_values(ascending = False)).index.values

sorted_data = data.reindex(new_index)
plt.figure(figsize = (15,10))

sns.barplot(x = sorted_data.area_list, y = sorted_data.area_poverty_ratio)

plt.xticks(rotation = 45)

plt.xlabel('States')

plt.ylabel('Poverty')

plt.title('Poverty Rate Given Of States')

plt.show()
PoliceKillingsUS.head()
PoliceKillingsUS.info()
seperate = PoliceKillingsUS.name[PoliceKillingsUS.name != 'TK TK'].str.split()

name, surname = zip(*seperate)

count = Counter(name + surname)

mostCommons = count.most_common(15)

mostCommonNames, frequency = zip(*mostCommons)

mostCommonNames, frequency = list(mostCommonNames), list(frequency)
plt.figure(figsize = (15, 10))

sns.barplot(x = mostCommonNames, y = frequency)
PercentOver25CompletedHighSchool.head()
PercentOver25CompletedHighSchool.info()
#PercentOver25CompletedHighSchool.percent_completed_hs.value_counts()

PercentOver25CompletedHighSchool.percent_completed_hs.replace(['-'], 0.0, inplace = True)

PercentOver25CompletedHighSchool.percent_completed_hs = PercentOver25CompletedHighSchool.percent_completed_hs.astype('float')
area_list = list(PercentOver25CompletedHighSchool['Geographic Area'].unique())

area_highschool = []



for i in area_list:

    x = PercentOver25CompletedHighSchool[PercentOver25CompletedHighSchool['Geographic Area'] == i]

    area_highschool_rate = sum(x.percent_completed_hs) / len(x)

    area_highschool.append(area_highschool_rate)
data = pd.DataFrame({'area_list': area_list, 'area_highschool_ratio': area_highschool})

new_index = (data.area_highschool_ratio.sort_values(ascending = False)).index.values

sorted_data2 = data.reindex(new_index)
plt.figure(figsize = (15,10))

sns.barplot(x = sorted_data2.area_list, y = sorted_data2.area_highschool_ratio)

plt.xticks(rotation = 45)

plt.xlabel('States')

plt.ylabel('High School Graduation Ratio')

plt.title('High school graduation rate of the population that is older than 25 in states')

plt.show()
ShareRaceByCity.head()
ShareRaceByCity.info()
ShareRaceByCity.replace(['(X)'], 0.0, inplace = True)
ShareRaceByCity.share_black = ShareRaceByCity.share_black.astype('float')

ShareRaceByCity.share_white = ShareRaceByCity.share_white.astype('float')

ShareRaceByCity.share_native_american = ShareRaceByCity.share_native_american.astype('float')

ShareRaceByCity.share_asian = ShareRaceByCity.share_asian.astype('float')

ShareRaceByCity.share_hispanic = ShareRaceByCity.share_hispanic.astype('float')
area_list = list(ShareRaceByCity['Geographic area'].unique())

share_black = []

share_white = []

share_native_american = []

share_asian = []

share_hispanic = []



for i in area_list:

    x = ShareRaceByCity[ShareRaceByCity['Geographic area'] == i]

    share_black.append(sum(x.share_black) / len(x))

    share_white.append(sum(x.share_white) / len(x))

    share_native_american.append(sum(x.share_native_american) / len(x))

    share_asian.append(sum(x.share_asian) / len(x))

    share_hispanic.append(sum(x.share_hispanic) / len(x))
f, ax = plt.subplots(figsize = (9, 15))

sns.barplot(x = share_black, y = area_list, color = 'blue', alpha = 0.7, label = 'Black')

sns.barplot(x = share_white, y = area_list, color = 'green', alpha = 0.5, label = 'White')

sns.barplot(x = share_native_american, y = area_list, color = 'cyan', alpha = 0.6, label = 'Native American')

sns.barplot(x = share_asian, y = area_list, color = 'yellow', alpha = 0.6, label = 'Asian')

sns.barplot(x = share_hispanic, y = area_list, color = 'red', alpha = 0.6, label = 'Hispanic')



ax.legend(loc = 'lower right', frameon = True)

ax.set(xlabel = 'Race Ratios', ylabel = 'States', title = 'Race ratios of states')

sorted_data2.head()
sorted_data.area_poverty_ratio = sorted_data.area_poverty_ratio / max(sorted_data.area_poverty_ratio)

sorted_data2.area_highschool_ratio = sorted_data2.area_highschool_ratio / max(sorted_data2.area_highschool_ratio)

data = pd.concat([sorted_data, sorted_data2.area_highschool_ratio], axis = 1)

data.sort_values('area_poverty_ratio', inplace = True)
f, ax =  plt.subplots(figsize = (20,10))

sns.pointplot(x = 'area_list', y = 'area_poverty_ratio', data = data, color = 'lime', alpha = 0.8)

sns.pointplot(x = 'area_list', y = 'area_highschool_ratio', data = data, color = 'red', alpha = 0.8)

plt.text(35,0.6, 'High school graduation ratio', color = 'red', fontsize = 17, style = 'italic')

plt.text(35,0.55, 'Poverty ratio', color = 'lime', fontsize = 17, style = 'italic')

plt.grid()

ax.set(xlabel = 'States', ylabel = 'Values', title = 'High school graduation vs poverty rate of states')
sns.kdeplot(data.area_poverty_ratio, data.area_highschool_ratio, shade = True, cut = 3)

plt.show()
sns.jointplot(data.area_poverty_ratio, data.area_highschool_ratio, kind = 'scatter', ratio = 3, color = 'g')
sns.lmplot(x = 'area_poverty_ratio', y = 'area_highschool_ratio', data = data)

plt.show()
sns.violinplot(data = data, inner = 'points')

plt.show()
f, ax = plt.subplots(figsize = (5, 5))

sns.heatmap(data.corr(), annot = True, linewidths = .5, fmt = '.2f', ax = ax)

plt.show()
sns.pairplot(data)

plt.show()
PoliceKillingsUS.race.dropna(inplace = True)
labels = PoliceKillingsUS.race.value_counts().index

colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral', 'cyan', 'grey']

explode = [0,0,0,0,0,0]

sizes = PoliceKillingsUS.race.value_counts().values
plt.figure(figsize = (7,7))

plt.pie(sizes, explode = explode , labels = labels, colors = colors, autopct = '%1.2f%%')

plt.title('Race rates according to kill data', size = 17)
sns.boxplot(x = 'gender', y = 'age', hue = 'manner_of_death', data = PoliceKillingsUS)

plt.show()
sns.swarmplot(x = 'gender', y = 'age', hue = 'manner_of_death' ,data = PoliceKillingsUS)

plt.show()
sns.countplot(PoliceKillingsUS.gender)

plt.title('Gender', fontsize = 15)

plt.show()
armed =PoliceKillingsUS.armed.value_counts()

plt.figure(figsize = (20, 7))

sns.barplot(x = armed[:14].index, y = armed[:14].values)

plt.title('Weapon count per death')

plt.ylabel('Count')

plt.xlabel('Weapon')

plt.show()
ages = ['above25' if i > 25 else '25' if i == 25 else 'below25' for i in PoliceKillingsUS.age]

df = pd.DataFrame({'age' : ages})

sns.countplot(x = df.age)

plt.show()
sns.countplot(data = PoliceKillingsUS, x = 'race')
city = PoliceKillingsUS.city.value_counts()

plt.figure(figsize = (15, 7))

sns.barplot(x = city[:12].index, y = city[:12].values)

states = PoliceKillingsUS.state.value_counts()

plt.figure(figsize = (15, 7))

sns.barplot(x = states[:12].index, y = states[:12].values)

plt.show()
sns.countplot(PoliceKillingsUS.signs_of_mental_illness)

plt.ylabel('Number of mental illness')

plt.xlabel('Mental illness')

plt.show()
sns.countplot(PoliceKillingsUS.threat_level)

plt.xlabel('Threat Types')

plt.show()
sns.countplot(PoliceKillingsUS.flee)

plt.xlabel('Flee Types')

plt.show()