#importing Libraries



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from collections import Counter

%matplotlib inline

#Importing data files in .csv format



percent_over_25_completed_highSchool = pd.read_csv('../input/input-files/PercentOver25CompletedHighSchool (1).csv',encoding="windows-1252")

percentage_people_below_poverty_level = pd.read_csv('../input/input-files/PercentagePeopleBelowPovertyLevel (1).csv', encoding="windows-1252")

kill = pd.read_csv('../input/input-files/PoliceKillingsUS (1).csv', encoding="windows-1252")

share_race_city = pd.read_csv('../input/input-files/ShareRaceByCity _name.csv', encoding="windows-1252")

median_house_hold_in_come = pd.read_csv('../input/input-files/datasets_2647_4395_MedianHouseholdIncome2015 (1).csv', encoding="windows-1252")

#Checking the data type



percentage_people_below_poverty_level.info()
# Changing data type from object to float



percentage_people_below_poverty_level.poverty_rate.replace(['-'],0.0,inplace = True)

percentage_people_below_poverty_level.poverty_rate = percentage_people_below_poverty_level.poverty_rate.astype(float)

percentage_people_below_poverty_level.poverty_rate.describe()
## The comparison of the poverty rate among all States



percentage_people_below_poverty_level.poverty_rate.replace(['-'],0.0,inplace = True)

percentage_people_below_poverty_level.poverty_rate = percentage_people_below_poverty_level.poverty_rate.astype(float)



area_list = list(percentage_people_below_poverty_level['Geographic Area'].unique())

area_poverty_ratio = []

for i  in area_list:

    x = percentage_people_below_poverty_level[percentage_people_below_poverty_level['Geographic Area']==i]

    area_poverty_rate = sum(x.poverty_rate)/len(x)

    area_poverty_ratio.append(area_poverty_rate)

data = pd.DataFrame({'area_list': area_list,'area_poverty_ratio':area_poverty_ratio})

new_index = (data['area_poverty_ratio'].sort_values(ascending=False)).index.values

sorted_data = data.reindex(new_index)



# Respresenting the data in visual form



plt.figure(figsize=(15,10))

sns.barplot(x=sorted_data['area_list'], y=sorted_data['area_poverty_ratio'])

plt.xticks(rotation= 45)

plt.xlabel('States')

plt.ylabel('Poverty Rate')

plt.title('Poverty Rate Given States')
#Checking the data type



percent_over_25_completed_highSchool.info()
# High school graduation rate of the population that is older than 25 in states



percent_over_25_completed_highSchool.percent_completed_hs.replace(['-'],0.0,inplace = True)

percent_over_25_completed_highSchool.percent_completed_hs = percent_over_25_completed_highSchool.percent_completed_hs.astype(float)

area_list = list(percent_over_25_completed_highSchool['Geographic Area'].unique())

area_highschool = []

for i in area_list:

    x = percent_over_25_completed_highSchool[percent_over_25_completed_highSchool['Geographic Area']==i]

    area_highschool_rate = sum(x.percent_completed_hs)/len(x)

    area_highschool.append(area_highschool_rate)

# sorting

data = pd.DataFrame({'area_list': area_list,'area_highschool_ratio':area_highschool})

new_index = (data['area_highschool_ratio'].sort_values(ascending=True)).index.values

sorted_data2 = data.reindex(new_index)



# Respresenting the data in visual form



plt.figure(figsize=(15,10))

sns.barplot(x=sorted_data2['area_list'], y=sorted_data2['area_highschool_ratio'])

plt.xticks(rotation= 90)

plt.xlabel('States')

plt.ylabel('High School Graduate Rate')

plt.title("Percentage of Given State's Population Above 25 that Has Graduated High School")
# Finding the state with the highest no of kills and plotting the data on the barplot



state = kill.state.value_counts()

plt.figure(figsize=(10,7))

sns.barplot(x=state[:20].index,y=state[:20].values)

plt.title('Most dangerous state',color = 'red',fontsize=18)
# Finding the city with the highest no of kills and plotting the data on the barplot



city = kill.city.value_counts()

plt.figure(figsize=(10,7))

sns.barplot(x=city[:10].index,y=city[:10].values)

plt.xticks(rotation=45)

plt.title('Most dangerous cities',color = 'orange',fontsize=18)
# Representing the kill data through graphical representation



cat_var=['manner_of_death','gender','signs_of_mental_illness','threat_level']

fig, ax = plt.subplots(1,4, figsize=(14,6))

for var, subplot in zip(cat_var, ax.flatten()):

   kill[var] .value_counts().plot(kind='bar', ax=subplot, title=var)

fig.tight_layout()

# Finding the Kill weapon



armed = kill.armed.value_counts()

#print(armed)

plt.figure(figsize=(10,7))

sns.barplot(x=armed[:7].index,y=armed[:7].values)

plt.ylabel('Number of Weapon')

plt.xlabel('Weapon Types')

plt.title('Kill weapon',color = 'blue',fontsize=15)
# age of killed people



above25 =['above25' if i >= 25 else 'below25' for i in kill.age]

df = pd.DataFrame({'age':above25})

sns.countplot(x=df.age)

plt.ylabel('Number of Killed People')

plt.title('Age of killed people',color = 'blue',fontsize=15)
# Creating a countplot for visual representation of Flee Types



sns.countplot(kill.flee)

plt.xlabel('Flee Types')

plt.title('Flee types',color = 'blue', fontsize = 15)
# Did the Police have body cameras on them ?



sns.countplot(kill.body_camera)

plt.xlabel('Having Body Cameras')

plt.title('Did the Police have body cameras on them ?',color = 'blue',fontsize = 15)
# Converting data type from object to float



num_var=['share_white','share_black','share_native_american','share_asian','share_hispanic']

share_race_city.replace(['-'],0.0,inplace = True)

share_race_city.replace(['(X)'],0.0,inplace = True)

share_race_city.loc[:,['num_var']] = share_race_city.loc[:,['share_white','share_black','share_native_american','share_asian','share_hispanic']].astype(float)
# Finding brief statistics about the population share of the races



share_race_city.describe()
# Percentage of states population according to races which namely are white,african american,native american,asian and hispanic



share_race_city.replace(['-'],0.0,inplace = True)

share_race_city.replace(['(X)'],0.0,inplace = True)

share_race_city.loc[:,['share_white','share_black','share_native_american','share_asian','share_hispanic']] = share_race_city.loc[:,['share_white','share_black','share_native_american','share_asian','share_hispanic']].astype(float)

area_list = list(share_race_city['Geographic area'].unique())

share_white = []

share_black = []

share_native_american = []

share_asian = []

share_hispanic = []

for i in area_list:

    x = share_race_city[share_race_city['Geographic area']==i]

    share_white.append(sum(x.share_white)/len(x))

    share_black.append(sum(x.share_black) / len(x))

    share_native_american.append(sum(x.share_native_american) / len(x))

    share_asian.append(sum(x.share_asian) / len(x))

    share_hispanic.append(sum(x.share_hispanic) / len(x))



# Respresenting the data in visual form



f,ax = plt.subplots(figsize = (9,15))

sns.barplot(x=share_white,y=area_list,color='green',alpha = 0.5,label='White' )

sns.barplot(x=share_black,y=area_list,color='blue',alpha = 0.7,label='African American')

sns.barplot(x=share_native_american,y=area_list,color='cyan',alpha = 0.6,label='Native American')

sns.barplot(x=share_asian,y=area_list,color='yellow',alpha = 0.6,label='Asian')

sns.barplot(x=share_hispanic,y=area_list,color='red',alpha = 0.6,label='Hispanic')



ax.legend(loc='lower right',frameon = True) 

ax.set(xlabel='Percentage of Races', ylabel='States',title = "Percentage of State's Population According to Races ")

# Representing population share of races through a histogram



share_race_city[num_var].hist(edgecolor='black', bins=10, figsize=(14, 5), layout = (2,3));
# Creating a piechart for people killed according to races



kill.race.dropna(inplace = True)

labels = kill.race.value_counts().index

colors = ['grey','blue','red','yellow','green','brown']

explode = [0,0,0,0,0,0]

sizes = kill.race.value_counts().values



# visual

plt.figure(figsize = (7,7))

plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%')

plt.title('Killed People According to Races',color = 'blue',fontsize = 15)