# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter

%matplotlib inline
income = pd.read_csv('../input/MedianHouseholdIncome2015.csv', encoding='ISO-8859-1')
poverty = pd.read_csv('../input/PercentagePeopleBelowPovertyLevel.csv', encoding='ISO-8859-1')
high_school = pd.read_csv('../input/PercentOver25CompletedHighSchool.csv', encoding='ISO-8859-1')
race = pd.read_csv('../input/ShareRaceByCity.csv', encoding='ISO-8859-1')
kill = pd.read_csv('../input/PoliceKillingsUS.csv', encoding='ISO-8859-1')
# values = poverty['poverty_rate'].unique()
# values.sort()
# print(values)

poverty.replace(['-'], 0.0, inplace=True)
poverty['poverty_rate'] = poverty['poverty_rate'].astype(float)

area_list = poverty['Geographic Area'].unique()
area_poverty_ratio = []
for i in area_list:
    x = poverty[poverty['Geographic Area'] == i]
    rate = sum(x['poverty_rate']) / len(x)
    area_poverty_ratio.append(rate)

data = pd.DataFrame({
    'area_list': area_list,
    'area_poverty_ratio': area_poverty_ratio
})

new_index = (data['area_poverty_ratio'].sort_values(ascending=False)).index.values
sorted_data = data.reindex(new_index)

plt.figure(figsize=(15, 10))
ax = sns.barplot(x=sorted_data['area_list'], y=sorted_data['area_poverty_ratio'])
plt.xticks(rotation=90)
plt.xlabel('States')
plt.ylabel('Poverty Rate')
plt.title('Poverty Rate Given States')


separate = kill[kill['name'] != 'TK TK']['name'].str.split()

a, b = zip(*separate)
name_list = a + b
name_count = Counter(name_list)
most_common_names = name_count.most_common(15)
x, y = zip(*most_common_names)

plt.figure(figsize=(15, 10))
ax = sns.barplot(x=x, y=y, palette=sns.cubehelix_palette(len(x)))
plt.xlabel('Name or Surname of killed people')
plt.ylabel('Frequency')
plt.title('Most common 15 Name or Surname of killed people')
high_school['percent_completed_hs'].replace(['-'], 0.0, inplace=True)
high_school['percent_completed_hs'] = high_school['percent_completed_hs'].astype(float)
high_school_data = high_school.groupby('Geographic Area')['percent_completed_hs'].mean().sort_values(ascending=True).to_frame()

plt.figure(figsize=(15, 10))
ax = sns.barplot(x=data.index, y=data['percent_completed_hs'])
plt.xticks(rotation=90)
plt.xlabel('States')
plt.ylabel('High School Graduate Rate')
plt.title('Percentage of Given State Population ABove 25 that Has Graduated High School')
race_list = ['share_white', 'share_black', 'share_native_american', 'share_asian', 'share_hispanic']
race.replace(['-', '(X)'], 0.0, inplace=True)
race.loc[:, race_list] = race.loc[:, race_list].astype(float)

white = race.groupby('Geographic area')['share_white'].mean()
black = race.groupby('Geographic area')['share_black'].mean()
native_american = race.groupby('Geographic area')['share_native_american'].mean()
asian = race.groupby('Geographic area')['share_asian'].mean()
hispanic = race.groupby('Geographic area')['share_hispanic'].mean()

f, ax = plt.subplots(figsize=(9, 15))
sns.barplot(x=white, y=white.index, color='green', alpha=0.5, label='White')
sns.barplot(x=black, y=black.index, color='blue', alpha=0.7, label='Latin American')
sns.barplot(x=native_american, y=native_american.index, color='cyan', alpha=0.6, label='Native American')
sns.barplot(x=asian, y=asian.index, color='yellow', alpha=0.6, label='Asian')
sns.barplot(x=hispanic, y=hispanic.index, color='red', alpha=0.6, label='Hispanic')

ax.legend(loc='lower right', frameon=True)
ax.set_xlabel('Percentage of Races')
ax.set_ylabel('States')
ax.set_title('Percentage of States Population According to Races')
sorted_data['area_poverty_ratio'] /= max(sorted_data['area_poverty_ratio'])
high_school_data['percent_completed_hs'] /= max(high_school_data['percent_completed_hs'])

# data = pd.concat([sorted_data, high_school_data['percent_completed_hs']], axis=1)
data = sorted_data.join(high_school_data, on='area_list')
data.sort_values('area_poverty_ratio', inplace=True)

f, ax = plt.subplots(figsize=(20, 10))
sns.pointplot(x='area_list', y='area_poverty_ratio', data=data, color='lime', alpha=0.8)
sns.pointplot(x='area_list', y='percent_completed_hs', data=data, color='red', alpha=0.8)
plt.text(40, 0.6, 'high school graduate ratio', color='red', fontsize=17, style='italic')
plt.text(40, 0.55, 'poverty ratio', color='lime', fontsize=18, style='italic')
plt.xlabel('States', fontsize=15, color='blue')
plt.ylabel('Values', fontsize=15, color='blue')
plt.title('High School Graduate vs Poverty rate', fontsize=20, color='blue')
plt.grid()
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(7, 5))

sns.countplot(kill['gender'], ax=ax1)
sns.countplot(kill['manner_of_death'], ax=ax2)
armed = kill.armed.value_counts()
plt.figure(figsize=(10, 7))
sns.barplot(x=armed[:7].index, y=armed[:7].values)
plt.ylabel('Number of weapon')
plt.xlabel('Weapon types')
plt.title('Kill weapon', color='blue', fontsize=15)
above25 = ['above25' if i >= 25 else 'below25' for i in kill['age']]
df = pd.DataFrame({ 'age': above25 })
sns.countplot(x=df.age)
plt.ylabel('Number of killed people')
plt.title('Age of killed people', color='blue', fontsize=15)
sns.countplot(data=kill, x='race')
plt.title('Race of killed people', color='blue', fontsize=15)
city = kill['city'].value_counts()
plt.figure(figsize=(10, 7))
sns.barplot(x=city[:12].index, y=city[:12].values)
plt.xticks(rotation=45)
plt.title('Most dangerous cities', color='blue', fontsize=15)
state = kill.state.value_counts()
plt.figure(figsize=(10, 7))
sns.barplot(x=state[:20].index, y=state[:20].values)
plt.title('Most dangerous state', color='blue', fontsize=15)
kill['race'].dropna(inplace=True)
labels = kill['race'].value_counts().index
colors = ['grey', 'blue', 'red', 'yellow', 'green', 'brown']
explode = [0, 0, 0, 0, 0, 0]
sizes = kill['race'].value_counts().values
plt.figure(figsize=(10, 10))
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%')
plt.title('Killed people according to races', color='blue', fontsize=15)

