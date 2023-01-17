import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import os

import warnings

from collections import Counter



warnings.filterwarnings("ignore")

print(os.listdir("../input"))
average_income = pd.read_csv('../input/MedianHouseholdIncome2015.csv', encoding="windows-1252")

poverty_level = pd.read_csv('../input/PercentagePeopleBelowPovertyLevel.csv', encoding="windows-1252")

completed_highschool = pd.read_csv('../input/PercentOver25CompletedHighSchool.csv', encoding="windows-1252")

police_killings = pd.read_csv('../input/PoliceKillingsUS.csv', encoding="windows-1252")

killed_people = pd.read_csv('../input/ShareRaceByCity.csv', encoding="windows-1252")
average_income.head()
killed_people.head()
completed_highschool.head()
police_killings.head()
poverty_level.head()
poverty_level.info()
poverty_level['Geographic Area'].unique()
poverty_level.poverty_rate.replace(['-'], 0.0, inplace = True)

poverty_level.poverty_rate = poverty_level.poverty_rate.astype(float)

city_list = list(poverty_level['Geographic Area'].unique())

cities_poverty_rate = []

for each in city_list:

    tmp = poverty_level[poverty_level['Geographic Area'] == each]

    average = sum(tmp.poverty_rate) / len(tmp)

    cities_poverty_rate.append(average)



#Filtering

data_Frame = pd.DataFrame({'city': city_list, 'poverty_rate': cities_poverty_rate})

newIndex = (data_Frame.poverty_rate.sort_values(ascending = False)).index.values

sorted_data = data_Frame.reindex(newIndex)



#visualization

plt.figure(figsize = (15,10))

sns.barplot(x = sorted_data['city'], y = sorted_data['poverty_rate'])

plt.xticks(rotation = 45)

plt.xlabel('Cities')

plt.ylabel('Poverty Rate')

plt.title('Cities Poverty Rate of Cities')

plt.show()
police_killings.info()
# Most common 20 Name of killed people



seperate = police_killings.name[police_killings.name != 'TK TK'].str.split()

x,y = zip(*seperate)

name_list = x

name_count = Counter(name_list)

most_name = name_count.most_common(20)

a,b = zip(*most_name)

a, b = list(a), list(b)



#visualization

plt.figure(figsize = (15,10))

sns.barplot(x = a, y = b, palette = sns.cubehelix_palette(len(a)))

plt.xlabel('Names')

plt.ylabel('Frequency')

plt.title('Most common 15 Name of killed people')
killed_people.head()
killed_people.info()
killed_people.replace(['-'], 0.0, inplace = True)

killed_people.replace(['(X)'], 0.0, inplace= True)

killed_people.loc[:,['share_white','share_black','share_native_american', 'share_asian', 'share_hispanic']] = killed_people.loc[:,['share_white','share_black','share_native_american', 'share_asian', 'share_hispanic']].astype(float)

city_list = list(killed_people['Geographic area'].unique())



white_rate = []

black_rate = []

native_rate = []

asian_rate = []

hispanic_rate = []



for each in city_list:

    tmp = killed_people[killed_people['Geographic area'] == each]

    white_rate.append(sum(tmp.share_white) / len(tmp))

    black_rate.append(sum(tmp.share_black) / len(tmp))

    native_rate.append(sum(tmp.share_native_american) / len(tmp))

    asian_rate.append(sum(tmp.share_asian) / len(tmp))

    hispanic_rate.append(sum(tmp.share_hispanic) / len(tmp))

    

    

#visualization

f,vs = plt.subplots(figsize = (10, 15))

sns.barplot(x = white_rate, y = city_list, color = 'grey', alpha = 0.5, label = 'White American')

sns.barplot(x = black_rate, y = city_list, color = 'brown', alpha = 0.5, label = 'African American')

sns.barplot(x = native_rate, y = city_list, color = 'red', alpha = 0.5, label = 'Native American')

sns.barplot(x = asian_rate, y = city_list, color = 'aqua', alpha = 0.5, label = 'Asian American')

sns.barplot(x = hispanic_rate, y = city_list, color = 'blue', alpha = 0.5, label = 'Hispanic American')

vs.legend(loc = 'lower right', frameon = True)

vs.set(xlabel = 'Cities', ylabel = 'Race of People', title = ' The percentage of States Population According to Races')
completed_highschool.head()
completed_highschool.info()
# high school graduation rate vs Poverty rate of each state

sorted_data.poverty_rate = sorted_data.poverty_rate / max(sorted_data.poverty_rate)



completed_highschool.percent_completed_hs.replace(['-'], 0.0, inplace = True)

completed_highschool.percent_completed_hs = completed_highschool.percent_completed_hs.astype(float)

city_list = list(completed_highschool['Geographic Area'].unique())

completed_rate = []

for each in city_list:

    tmp = completed_highschool[completed_highschool['Geographic Area'] == each]

    average = sum(tmp.percent_completed_hs) / len(tmp)

    completed_rate.append(average)



dataFrame = pd.DataFrame({'city': city_list, 'rate': completed_rate})

newIndex = (dataFrame.rate.sort_values(ascending = False)).index.values

sorted_data2 = dataFrame.reindex(newIndex)

sorted_data2.rate = sorted_data2.rate / max(sorted_data2.rate)



merge_data = pd.concat([sorted_data, sorted_data2['rate']], axis = 1)

merge_data.sort_values('poverty_rate', inplace = True)



# visualization

f, vs1 = plt.subplots(figsize = (18,10))

sns.pointplot(x = 'city', y = 'poverty_rate', data = merge_data, color = 'r', alpha = 0.7)

sns.pointplot(x = 'city', y = 'rate', data = merge_data, color = 'b', alpha = 0.7)

plt.text(37, 0.5, 'High School Graduate Ratio', color = 'red', size = 18)

plt.text(40, 0.45, 'Poverty Ratio', color = 'blue', size = 18)

plt.xlabel('Poverty Rate', fontsize = 18, color = 'blue')

plt.ylabel('Graduate Rate', fontsize = 18, color = 'blue')

plt.title('High School Graduation Rate vs Poverty Rate', color = 'blue', fontsize = 20)

plt.grid()
# The same graphic with joint plot



sns.jointplot(merge_data.poverty_rate, merge_data.rate, kind = 'kde', size = 7)

plt.savefig('graph.png')

plt.show()
police_killings.race.value_counts()
# Race rates  



police_killings.race.dropna(inplace = True)

labels = police_killings.race.value_counts().index

colors = ['red', 'blue', 'brown', 'grey','green','aqua']

explode = [0,0,0,0,0,0]

values = police_killings.race.value_counts().values



#visualization

plt.figure(figsize = (9, 9))

plt.pie(values, explode = explode, labels = labels, colors = colors, autopct = '%1.1f%%')

plt.title('Killed People According to Races', color = 'blue', size = 20)

plt.show()
#The same graphic with lmplot



sns.lmplot(x = 'poverty_rate', y = 'rate', data = merge_data)

plt.show()
# The same table with kde plot

# cut = size of the graphic

# shade = whether or not inside of the grafic is fill

sns.kdeplot(merge_data.poverty_rate, merge_data.rate, color = 'green', shade = True, cut = 5)

plt.show()
# The same table with violin plot



pal = sns.cubehelix_palette(2, rot=-.5, dark=.3)

sns.violinplot(data= merge_data, palette = pal, inner = 'point')

plt.show()
# The same table with heatmap to see their correlation



f,vs = plt.subplots(figsize = (8,7))

sns.heatmap(merge_data.corr(), annot = True, fmt = '.1f', ax = vs)

plt.show()
police_killings.head()
sns.boxplot(x = 'gender', y = 'age', hue = 'manner_of_death', data = police_killings, linewidth=1.5, palette="hls")

plt.show()
#Swarm Plot 



sns.swarmplot(x = 'gender', y = 'age', hue = 'manner_of_death', data = police_killings)

plt.show()
#Pair Plot with same example

sns.pairplot(merge_data)

plt.show()
police_killings.head()
sns.countplot(police_killings.threat_level)

plt.title('The dates of Killings', fontsize = 15, color='red')
armed = police_killings.armed.value_counts()



plt.figure(figsize = (15,10))

sns.barplot(x = armed[:10].index, y = armed[:10].values)

plt.title('The Type of Kill Weapon', fontsize = 20, color = 'red')

plt.xlabel('Weapons')

plt.ylabel('Counts')

filtred_data = ['above 25' if each >= 20 else 'below 25' for each in police_killings.age]

df = pd.DataFrame({'age' : filtred_data})



sns.countplot(x = df.age)

plt.title('The Age of Killed People')

plt.show()
sns.countplot(data = police_killings, x = 'race')

plt.title('The Race of Killed People')

plt.show()
#Most Dangerous Cities

cities = police_killings.city.value_counts()

plt.figure(figsize = (10,6))

sns.barplot(x = cities[0:10].index, y = cities.values[:10])

plt.title('Most Dangerous Cities', fontsize = 20, color = 'red')
# Having mental ilness or not for killed people

plt.figure(figsize = (7,5))

sns.countplot(police_killings.signs_of_mental_illness)

plt.xlabel('Mental Ilness')

plt.ylabel('The number of mental ilness')

plt.title('Having mental ilness or not')
# Kill numbers from states in kill data

state = police_killings.state.value_counts().index[:10]

sns.barplot(x = state, y = police_killings.state.value_counts().values[:10])

plt.title('Kill numbers from states', fontsize = 20, color = 'r')