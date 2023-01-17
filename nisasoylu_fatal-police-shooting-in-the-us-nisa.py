# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from collections import Counter



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
below_poverty = pd.read_csv("/kaggle/input/fatal-police-shootings-in-the-us/PercentagePeopleBelowPovertyLevel.csv", encoding = "windows-1252")

police_killings = pd.read_csv("/kaggle/input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv", encoding = "windows-1252")

share_race_city = pd.read_csv("/kaggle/input/fatal-police-shootings-in-the-us/ShareRaceByCity.csv", encoding = "windows-1252")

completed_high_school_over25 = pd.read_csv("/kaggle/input/fatal-police-shootings-in-the-us/PercentOver25CompletedHighSchool.csv", encoding = "windows-1252")

house_income_median = pd.read_csv("/kaggle/input/fatal-police-shootings-in-the-us/MedianHouseholdIncome2015.csv", encoding = "windows-1252")
below_poverty.head(10)
below_poverty.info()
below_poverty.isnull().any()
below_poverty.poverty_rate.value_counts()
below_poverty.poverty_rate.replace("-",0, inplace = True)
below_poverty.poverty_rate.value_counts()
below_poverty.poverty_rate = below_poverty.poverty_rate.astype(float)
below_poverty.poverty_rate.value_counts()
below_poverty["Geographic Area"].value_counts().index
below_poverty["Geographic Area"].value_counts().values
below_poverty["Geographic Area"].unique()
unique_area_list = list(below_poverty["Geographic Area"].unique())
unique_area_list
plt.figure(figsize = (15,10))

plt.title("Poverty Level of Each State")

plt.bar(unique_area_list, below_poverty["Geographic Area"].value_counts().values) 

plt.xlabel("States")

plt.ylabel("Poverty Level")

plt.show()
plt.figure(figsize = (15,10))

plt.title("Poverty Level of Each State")

sns.barplot(unique_area_list, below_poverty["Geographic Area"].value_counts().values) 

plt.xticks(rotation = 45)   # xticks method is used to determinethe location of variable names on the x axis.

plt.xlabel("States")

plt.ylabel("Poverty Level")

plt.show()
police_killings.head(10)
police_killings.name.value_counts()
a = "nisa soylu"

a.split()
b = ["nisa soylu", "mine soylu","zeynep bumin soylu"]

c = []

for i in b:

    c.append(i.split())

    unzipped_version = zip(*c)

    print(list(unzipped_version))
first_part, second_part = zip(*(police_killings.name[police_killings.name != "TK TK"].str.split()))
print(first_part)
len(first_part)
print(second_part)
len(second_part)
name_list = first_part + second_part
print(name_list)
len(name_list)
name_count = Counter(name_list)
print(name_count)
most_used_15_names = name_count.most_common(15)
most_used_15_names
x,y = zip(*most_used_15_names)
x,y = list(x), list(y)
print(x)
print(y)
plt.figure(figsize = (15,10))

plt.bar(x,y)

plt.title("Most Common 15 Names or Surnames of Killed People")

plt.xlabel("Name or surname of killed people")

plt.ylabel("Frequency")

plt.show()
plt.figure(figsize = (15,10))

sns.barplot(x, y)

plt.xticks(rotation = 45) 

plt.xlabel("Name or surname of killed people")

plt.ylabel("Frequency")

plt.title("Most Common 15 Names or Surnames of Killed People")

plt.show()
completed_high_school_over25.head(10)
completed_high_school_over25.percent_completed_hs.value_counts()
completed_high_school_over25.percent_completed_hs.replace("-",0, inplace = True)
completed_high_school_over25.percent_completed_hs.value_counts()
completed_high_school_over25.info()
completed_high_school_over25.percent_completed_hs = completed_high_school_over25.percent_completed_hs.astype(float)
completed_high_school_over25.percent_completed_hs.value_counts()
list_of_states = completed_high_school_over25["Geographic Area"].unique()
type(list_of_states)
state_list = list(list_of_states)

state_list
#completed_high_school_over25["Geographic Area"].value_counts().index.sort_values()

completed_high_school_over25["Geographic Area"].sort_values().value_counts()

    
area_highschool = []

for i in state_list:

    x = completed_high_school_over25[completed_high_school_over25['Geographic Area']==i]

    area_highschool_rate = sum(x.percent_completed_hs)/len(x)

    area_highschool.append(area_highschool_rate)
data = pd.DataFrame({'area_list': state_list,'area_highschool_ratio':area_highschool})

new_index = (data['area_highschool_ratio'].sort_values(ascending=True)).index.values

sorted_data2 = data.reindex(new_index)


# visualization

plt.figure(figsize=(15,10))

sns.barplot(x=sorted_data2['area_list'], y=sorted_data2['area_highschool_ratio'])

plt.xticks(rotation= 45)

plt.xlabel('States')

plt.ylabel('High School Graduate Rate')

plt.title("Percentage of Given State's Population Above 25 that Has Graduated High School")

plt.show()
share_race_city.head()
share_race_city.info()
share_race_city["Geographic area"].value_counts()
share_race_city.City.value_counts()
share_race_city.share_white.value_counts()
share_race_city.share_black.value_counts()
share_race_city.share_native_american.value_counts()
share_race_city.share_asian.value_counts()
share_race_city.share_hispanic.value_counts()
share_race_city.replace("-",0, inplace = True)

share_race_city.replace("(X)",0, inplace = True)



share_race_city.share_white = share_race_city.share_white.astype(float)

share_race_city.share_black = share_race_city.share_black.astype(float)

share_race_city.share_native_american = share_race_city.share_native_american.astype(float)

share_race_city.share_asian = share_race_city.share_asian.astype(float)

share_race_city.share_hispanic = share_race_city.share_hispanic.astype(float)
share_race_city.info()
area_list = list(share_race_city["Geographic area"].unique())



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
plt.figure(figsize = (9,15))

plt.yticks(rotation = 0)

sns.barplot(share_white, area_list, color = "orange", label = "White")

sns.barplot(share_black, area_list, color = "brown", label = "African American")

sns.barplot(share_native_american, area_list, color = "red", label ="Native American")

sns.barplot(share_asian, area_list, color = "yellow", label ="Asian")

sns.barplot(share_hispanic, area_list, color = "purple", label ="Hispanic")

plt.xlabel("Percentage of Races")

plt.ylabel("States")

plt.title("Percentage Of State's Population According To Races")

plt.legend()

plt.show()
below_poverty.head()
below_poverty["Geographic Area"].value_counts()
below_poverty["poverty_rate"].value_counts()
city_names = list(below_poverty["Geographic Area"].unique())



poverty_rates_each_city = []



for i in city_names:

    x = below_poverty[below_poverty["Geographic Area"] == i]

    poverty_rates_each_city.append((sum(x.poverty_rate))/len(x))
poverty_rates_each_city
completed_high_school_over25.head()
graduation_rates_each_city = []



for i in city_names:

    x = completed_high_school_over25[completed_high_school_over25["Geographic Area"] == i]

    graduation_rates_each_city.append(sum(x.percent_completed_hs)/len(x))
graduation_rates_each_city
plt.figure(figsize = (20,10))

sns.pointplot(city_names, graduation_rates_each_city, color ="lime", label = "graduation rate")

sns.pointplot(city_names, poverty_rates_each_city,color = "brown", label = "poverty rate")



plt.xlabel('States',fontsize = 15,color='blue')

plt.ylabel('Values',fontsize = 15,color='blue')

plt.title('Percentage of High School Graduate  VS  Poverty Rate',fontsize = 20,color='blue')

plt.text(40,30.60,'high school graduate ratio',color='lime',fontsize = 17,style = 'italic')

plt.text(40,25.55,'poverty ratio',color='brown',fontsize = 18,style = 'italic')

plt.legend()

plt.grid()

plt.show()
sns.jointplot(poverty_rates_each_city, graduation_rates_each_city,ratio=3, color="purple", kind="kde", height=7)

plt.xlabel("area poverty percentage")

plt.ylabel("area highschool ratio")

plt.savefig('graph.png')

plt.show()
g = sns.jointplot(poverty_rates_each_city, graduation_rates_each_city, height=5, ratio=3, color="lime")

plt.xlabel("area poverty percentage")

plt.ylabel("area highschool ratio")

plt.show()
police_killings.head()
police_killings.race
police_killings.race.dropna(inplace = True)
police_killings.race
police_killings.race.value_counts()
list(police_killings.race.value_counts().index)
list(police_killings.race.value_counts().values)
plt.figure(figsize = (10,10))

colors = ["orange","purple","yellow","brown","lime","pink"]

explode = [0,0.1,0,0,0,0]  # only "explode" the 2nd slice (i.e. "B") 

plt.pie(list(police_killings.race.value_counts().values),explode,list(police_killings.race.value_counts().index),colors, autopct ="%1.1f%%")

plt.title("Killed People According to Races")

plt.show()
sns.kdeplot(poverty_rates_each_city, graduation_rates_each_city, color ="orange", shade = True) # shade => empty or filled shape

plt.xlabel("area poverty percentage")

plt.ylabel("area highschool ratio")

plt.show()
import numpy

n_poverty_rates_each_city = numpy.array(poverty_rates_each_city)

n_graduation_rates_each_city = numpy.array(graduation_rates_each_city)



plt.title("Poverty Percentage                 Graduation Percentage", color = "blue")



sns.violinplot(n_poverty_rates_each_city, color = "orange", inner = "points", label ="poverty percentage")

sns.violinplot(n_graduation_rates_each_city, color = "lime", inner = "points", label = "graduation percentage")

plt.legend()



plt.show()
n_poverty_rates_each_city/100
n_graduation_rates_each_city/100
sns.boxplot(police_killings.gender, police_killings. age, hue = police_killings.manner_of_death,data = police_killings)

plt.show()
police_killings.gender.unique()
police_killings.age.unique()
sns.swarmplot(police_killings.gender, police_killings.age, hue = police_killings.manner_of_death, data = police_killings)

plt.show()
police_killings.manner_of_death.value_counts()
sns.countplot(police_killings.gender)

plt.title("Gender", color = "orange")

plt.show()
police_killings.armed.value_counts().index
police_killings.armed.value_counts().values
plt.figure(figsize = (25,10))

sns.barplot(police_killings.armed.value_counts().index[:7], police_killings.armed.value_counts().values[:7])

plt.title("Gender", color = "lime")

plt.show()
sns.countplot(police_killings.manner_of_death)

plt.title("Gender", color = "red")

plt.show()