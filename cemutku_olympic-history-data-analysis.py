import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import os
import seaborn as sns
print(os.listdir("../input"))
# loading olympic data
data = pd.read_csv('../input/athlete_events.csv')
# information about dataset
data.info()
data.head(10)
data["Medal"].value_counts(dropna = False)
data["Sport"].value_counts(dropna = False)
# filter data for medals. We need data with medals
filter = ((data.Medal == 'Gold') | (data.Medal == 'Bronze') | (data.Medal == 'Silver'))
medal_data = data[filter]
#Total Medal Counts by Countries
countries = list(medal_data['NOC'].unique())
medal_counts = []
for i in countries:
    x = medal_data[medal_data['NOC'] == i]
    medalCount = len(x)
    medal_counts.append(medalCount)

newData = pd.DataFrame({'countries': countries,'medal_counts':medal_counts})
new_index = (newData['medal_counts'].sort_values(ascending=False)).index.values
sorted_data = newData.reindex(new_index)
#sorted_data = sorted_data[sorted_data["medal_counts"] > 100]

plt.figure(figsize = (30,10))
sns.barplot(x = sorted_data['countries'], y = sorted_data['medal_counts'])
plt.xticks(rotation = 90)
plt.xlabel('Countries')
plt.ylabel('Total Medal Count')
plt.title('Medal Counts by Countries')
plt.show()
# copy data for ages
ageData = data
ageData["Age"].dropna(inplace = True)
ageData["Age"].fillna(0 ,inplace = True)
# check null entries
ageData['Age'].notnull().all()
# getting sport types
sportTypes = list(ageData['Sport'].unique())
avarageAges = []
for i in sportTypes:
    x = ageData[ageData['Sport'] == i]
    x["Age"].dropna(inplace = True)
    x["Age"].fillna(0 ,inplace = True)
    avarageAge = sum(x.Age)/len(x)
    avarageAges.append(avarageAge)

sportAvarageAgeData = pd.DataFrame({'Sport' : sportTypes, 'AvarageAge' : avarageAges})
newSportAvarageAgeDataIndex = (sportAvarageAgeData['AvarageAge'].sort_values(ascending = False)).index.values
sportAvarageAgeDataSorted = sportAvarageAgeData.reindex(newSportAvarageAgeDataIndex)

plt.figure(figsize = (20,10))
sns.barplot(x = sportAvarageAgeDataSorted['Sport'], y = sportAvarageAgeDataSorted['AvarageAge'])
plt.xticks(rotation = 90)
plt.xlabel('Sport')
plt.ylabel('AvarageAge')
plt.title('AvarageAge by Sports')
plt.show()
#Genders by Games
games = list(medal_data['Games'].unique())
games.sort()
males = []
females = []
for i in games:
    m = medal_data[medal_data['Games'] == i]
    male = len(m[m.Sex == 'M'])
    males.append(male)
    female = len(m[m.Sex == 'F'])
    females.append(female)

# visualization
f,ax = plt.subplots(figsize = (10,15))
sns.barplot(x=males,y=games,color='black',alpha = 0.6,label='Male')
sns.barplot(x=females,y=games,color='yellow',alpha = 0.5,label='Female')
ax.legend(loc='lower right',frameon = True)
ax.set(xlabel='Sex', ylabel='Games',title = "Genders by Games")
plt.show()
# point plot
# Number of male/female athletes by years - Summer Games
summerData = data[data["Season"] == 'Summer']
years = list(summerData['Year'].unique())
femaleAthletes = []
maleAthletes = []

for i in years:
    x = summerData[summerData['Year'] == i]
    femaleCount = len(x[x['Sex'] == 'F'])
    femaleAthletes.append(femaleCount)
    maleCount = len(x[x['Sex'] == 'M'])    
    maleAthletes.append(maleCount)
    
athletesByYearsData = pd.DataFrame({'Year' : years, 'Male' : maleAthletes, 'Female' : femaleAthletes})
athletesByYearsDataIndex = (athletesByYearsData['Year'].sort_values(ascending = False)).index.values
athletesByYearsDataSorted = athletesByYearsData.reindex(athletesByYearsDataIndex)

f,ax1 = plt.subplots(figsize = (20,10))
sns.pointplot(x = 'Year', y = 'Male', data = athletesByYearsDataSorted, color = 'blue', alpha = 0.8)
sns.pointplot(x = 'Year', y = 'Female', data = athletesByYearsDataSorted, color = 'red', alpha = 0.8)
plt.text(2, 8000, 'Male', color = 'blue',fontsize = 17, style = 'italic')
plt.text(2, 8500, 'Female', color = 'red',fontsize = 18, style = 'italic')
plt.xlabel('Years', fontsize = 15, color = 'blue')
plt.ylabel('Count', fontsize = 15, color = 'blue')
plt.title('Number of male/female athletes by years - Summer Games', fontsize = 20, color = 'blue')
plt.grid()
athletesByYearsDataSorted.head()
# joint plot
# kind : { “scatter” | “reg” | “resid” | “kde” | “hex” }
sns.jointplot(athletesByYearsDataSorted.Year, athletesByYearsDataSorted.Female, kind = "kde", size = 7)
plt.savefig('graph.png')
plt.show()
# jointplot kind = scatter
sns.jointplot(athletesByYearsDataSorted.Year, athletesByYearsDataSorted.Female, kind = "scatter", size = 8, ratio = 3)
plt.show()
data.Medal.value_counts()
# pie chart
data2012 = data[data['Year'] == 2012]
data2012.Medal.value_counts()
labels = data2012.Medal.value_counts().index
colors = ['red', 'gold', 'silver']
explode = [0, 0, 0]
sizes = data2012.Medal.value_counts().values

plt.figure(figsize = (7, 7))
plt.pie(sizes, explode = explode, labels = labels, colors = colors , autopct='%1.1f%%')
plt.title('Medals according to 2012', color = 'blue', fontsize = 15)
plt.show()
# lmplot 
# Show the results of a linear regression within each dataset
sns.lmplot(x = "Year", y = "Female", data = athletesByYearsDataSorted)
plt.show()
sportAvarageAgeDataSorted.head()
# kdeplot
sns.kdeplot(athletesByYearsDataSorted.Year, athletesByYearsDataSorted.Female, shade = True, cut = 3)
plt.show()
# Show each distribution with both violins and points
# Use cubehelix to get a custom sequential palette
athletesByYearsDataSorted['Male'] = athletesByYearsDataSorted['Male']/max(athletesByYearsDataSorted['Male'])
athletesByYearsDataSorted['Female'] = athletesByYearsDataSorted['Female']/max(athletesByYearsDataSorted['Female'])
athletesByYearsDataSortedWithourYear = athletesByYearsDataSorted.drop('Year', 1)

pal = sns.cubehelix_palette(2, rot = - .5, dark = .3)
sns.violinplot(data = athletesByYearsDataSortedWithourYear, palette = pal, inner = "points")
plt.show()
# heatmap
data.corr()
f,ax = plt.subplots(figsize = (10, 10))
sns.heatmap(data.corr(), annot = True, linewidths = 0.5, linecolor = "red", fmt = '.1f', ax = ax)
plt.show()
#Height and Weight have positive correlation
# Plot the orbital period with horizontal boxes
plt.figure(figsize = (10, 10))
sns.boxplot(x = "Sex", y = "Year", hue = "Medal", data = data, palette = "PRGn")
plt.show()
# swarm plot
data2016 = data[data.Year == 2016]
plt.figure(figsize = (10, 10))
sns.swarmplot(x = "Sex", y = "Age", hue = "Medal", data = data2016)
plt.show()
# pair plot
sns.pairplot(athletesByYearsDataSorted)
plt.show()
# count plot
sns.countplot(data.Sex)
plt.title("Sex", color = 'blue', fontsize=15)
sorted_data.head()
# Top 10 sports
sports = data.Sport.value_counts()
plt.figure(figsize = (17, 7))
sns.barplot(x = sports[:10].index, y = sports[:10].values)
plt.ylabel('Number of Sport')
plt.xlabel('Sport Types')
plt.title('Top Sports', color = 'blue', fontsize=15)
above40 = ['above40' if i >= 40 else 'below40' for i in data.Age]
df = pd.DataFrame({'age' : above40})
sns.countplot(x = df.age)
plt.ylabel('Number of athletes')
plt.title('Age of athlethes', color = 'blue', fontsize = 15)
# select teams with gold medals
dataBasketballWithMedals = data[(data["Sport"] == "Basketball") & (data["Medal"] == "Gold")]
dataBasketballWithMedals.Team.value_counts()
plt.figure(figsize = (10, 7))
sns.countplot(x = dataBasketballWithMedals.Team)
plt.ylabel('Number of gold medals')
plt.title('Top Basketball Teams', color = 'blue', fontsize = 15)
# Top 5 Events
events = data["Event"].value_counts()
plt.figure(figsize = (20, 7))
sns.barplot(x = events[:5].index, y = events[:5].values)
plt.ylabel('Number of event')
plt.xlabel('Events')
plt.title('Top Events', color = 'blue', fontsize = 15)
plt.show()