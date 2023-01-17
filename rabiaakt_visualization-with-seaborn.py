# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import warnings

warnings.filterwarnings('ignore') 



from collections import Counter

    

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
median_house_hold_income = pd.read_csv("/kaggle/input/fatal-police-shootings-in-the-us/MedianHouseholdIncome2015.csv", encoding="windows-1252")

police_killings = pd.read_csv("/kaggle/input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv", encoding="windows-1252")

poverty_level = pd.read_csv("/kaggle/input/fatal-police-shootings-in-the-us/PercentagePeopleBelowPovertyLevel.csv", encoding="windows-1252")

percentage_over_high_schools = pd.read_csv("/kaggle/input/fatal-police-shootings-in-the-us/PercentOver25CompletedHighSchool.csv", encoding="windows-1252")

share_race_by_city = pd.read_csv("/kaggle/input/fatal-police-shootings-in-the-us/ShareRaceByCity.csv", encoding="windows-1252")
poverty_level.info()
poverty_level.head()
poverty_level.poverty_rate.value_counts()
poverty_level.poverty_rate.replace(['-'],0.0,inplace=True)
poverty_level.poverty_rate.value_counts()
poverty_level["Geographic Area"].unique()
poverty_level.poverty_rate = poverty_level.poverty_rate.astype(float)
poverty_level.poverty_rate.value_counts()
area_list = list(poverty_level["Geographic Area"].unique())

area_poverty_ratio = []

for i in area_list:

    x = poverty_level[poverty_level["Geographic Area"] == i]

    area_poverty_rate = sum(x.poverty_rate) / len(x)

    area_poverty_ratio.append(area_poverty_rate)

data = pd.DataFrame({"area_list" : area_list,"area_poverty_ratio" : area_poverty_ratio})

new_index = (data.area_poverty_ratio.sort_values(ascending = False)).index.values

sort_data = data.reindex(new_index) 



plt.figure(figsize=(15,10))

sns.barplot(x = sort_data["area_list"], y = sort_data["area_poverty_ratio"])

plt.xticks(rotation = 45)

plt.xlabel("States")

plt.ylabel("Poverty Ratio")

plt.show()
police_killings.head()
police_killings.name.value_counts()
seperate = police_killings.name[police_killings.name != "TK TK"].str.split()

a, b = zip(*seperate)

name = a+b

name_count = Counter(name)

#print(name_count)

most_common_name = name_count.most_common(15)

x, y = zip(*most_common_name)

x, y = list(x), list(y)

print(x)

print(y)

plt.figure(figsize=(15,10))

ax = sns.barplot(x = x, y=y, palette = sns.cubehelix_palette(len(x)))

plt.xlabel("Name or Surname of killed people")

plt.ylabel("Count")

plt.title("Most common 15 killed people")

plt.show()
percentage_over_high_schools.head()
percentage_over_high_schools.info()
percentage_over_high_schools.percent_completed_hs.value_counts()
percentage_over_high_schools.percent_completed_hs.replace(["-"], 0.0, inplace = True)

percentage_over_high_schools.percent_completed_hs = percentage_over_high_schools.percent_completed_hs.astype(float)

area_list = percentage_over_high_schools["Geographic Area"].unique()

area_highschool = []

for j in area_list:

    x = percentage_over_high_schools[percentage_over_high_schools["Geographic Area"] == j]

    area_highschool_rate = sum(x.percent_completed_hs) / len(x)

    area_highschool.append(area_highschool_rate)

print(area_highschool)

hs_data = pd.DataFrame({"area_list" : area_list, "area_hs_ratio" : area_highschool})

new_index1 = (hs_data["area_hs_ratio"].sort_values(ascending = True)).index.values

hs_data_sorted = hs_data.reindex(new_index1)





plt.figure(figsize=(15,10))

sns.barplot(x = hs_data_sorted["area_list"], y = hs_data_sorted["area_hs_ratio"])

plt.xticks(rotation = 45)

plt.xlabel("Areas")

plt.ylabel("Area highschool ratio")

plt.show()
share_race_by_city.head()
share_race_by_city.share_asian.value_counts()
share_race_by_city.info()
share_race_by_city.replace(["-"], 0.0 , inplace = True)

share_race_by_city.replace(['(X)'], 0.0 , inplace = True)

share_race_by_city.loc[:,['share_white','share_black', 'share_native_american', 'share_asian', 'share_hispanic' ]] = share_race_by_city.loc[:,['share_white','share_black', 'share_native_american', 'share_asian', 'share_hispanic' ]].astype(float)

area_list = list(share_race_by_city["Geographic area"].unique())

share_white = []

share_black = []

share_native_american = []

share_asian = []

share_hispanic = []



for i in area_list:

    x = share_race_by_city[share_race_by_city["Geographic area"] == i]

    share_white.append(sum(x.share_white) / len(x))

    share_black.append(sum(x.share_black) / len(x))

    share_native_american.append(sum(x.share_native_american) / len(x))

    share_asian.append(sum(x.share_asian) / len(x))

    share_hispanic.append(sum(x.share_hispanic) / len(x))

    

f, ax = plt.subplots(figsize=(9, 15))

sns.barplot(x = share_white , y = area_list, alpha = 0.5, label = "White", color = 'green' )

sns.barplot(x = share_black , y = area_list, alpha = 0.7, label = "Black", color = 'red' )

sns.barplot(x = share_native_american , y = area_list, alpha = 0.6, label = "Native American", color = 'blue' )

sns.barplot(x = share_asian , y = area_list, alpha = 0.6, label = "Asian", color = 'cyan' )

sns.barplot(x = share_hispanic , y = area_list, alpha = 0.6, label = "Hispanic", color = 'yellow' )



ax.legend(loc = "lower right", frameon = True)

ax.set(xlabel = "Percentage of Races", ylabel = "States", title = "Percentage of State's Population According to Races")







    
hs_data_sorted.head()
#high school graduation rate vs Poverty rate of each state

sort_data['area_poverty_ratio'] = sort_data['area_poverty_ratio'] / max(sort_data['area_poverty_ratio'])

hs_data_sorted["area_hs_ratio"] = hs_data_sorted["area_hs_ratio"] / max(hs_data_sorted["area_hs_ratio"])



data = pd.concat([sort_data, hs_data_sorted["area_hs_ratio"]], axis = 1 )

data.sort_values('area_poverty_ratio', inplace = True)



f, ax1 = plt.subplots(figsize = (20, 10))

sns.pointplot(x = 'area_list' , y = 'area_poverty_ratio', data = data, color = 'lime', alpha = 0.6)

sns.pointplot(x = 'area_list' , y = 'area_hs_ratio', data = data, color = 'cyan', alpha = 0.6)

plt.text(40, 0.6, 'hs graduate ratio', fontsize = 18, style = 'italic', color = 'cyan')

plt.text(40, 0.55, 'poverty ratio', fontsize = 17, style = 'italic', color = 'lime')

plt.title('High School Graduate  VS  Poverty Rate',fontsize = 20,color='blue')

plt.xlabel('States' , color = 'red')

plt.ylabel('High school graduate ratio vs poverty ratio')

plt.show()
# Visualization of high school graduation rate vs Poverty rate of each state with different style of seaborn code

#joint kernel density

#pearsonr= if it is 1, there is positive correlation and if it is, -1 there is negative correlation.

# If it is zero, there is no correlation between variables

g = sns.jointplot(data.area_poverty_ratio, data.area_hs_ratio, kind = 'kde', size = 7)

plt.savefig("jointplot.png")

plt.show()
g = sns.jointplot('area_poverty_ratio', 'area_hs_ratio', data = data, size = 5, ratio = 3, color = 'cyan')
police_killings.head()
police_killings.race.dropna(inplace = True)

labels = police_killings.race.value_counts().index

colors = ['grey', 'blue', 'red', 'lime', 'cyan', 'yellow']

explode = [0, 0, 0, 0, 0, 0]

sizes = police_killings.race.value_counts().values



plt.figure(figsize = (7,7))

plt.pie(sizes, explode = explode, labels = labels, colors = colors, autopct = '%1.1f%%')

plt.title("Killings by races pie plot", color = 'cyan', fontsize = 20)
data.head()
sns.lmplot(x = 'area_poverty_ratio', y = 'area_hs_ratio', data = data)

plt.show()
sns.kdeplot(data.area_poverty_ratio, data.area_hs_ratio, shade = True, cut = 5)

plt.show()
pal = sns.cubehelix_palette(2 , rot = -.5, dark = .3)

sns.violinplot(data = data, palette = pal, inner = 'points')

plt.show()
data.corr()
f, ax = plt.subplots(figsize = (5,5))

sns.heatmap(data.corr(), annot = True, linewidth = .5, fmt = '.11f', ax = ax, linecolor = 'cyan')

plt.show()
police_killings.head()
police_killings.manner_of_death.unique()
sns.boxplot(x = 'gender', y = 'age', hue = 'manner_of_death', data = police_killings, palette = 'PRGn')

plt.show()
sns.swarmplot(x = 'gender', y = 'age', hue = 'manner_of_death', data = police_killings)

plt.show()
sns.pairplot(data)

plt.show()
police_killings.head()
police_killings.manner_of_death.value_counts()
#sns.countplot(police_killings.gender)

sns.countplot(police_killings.manner_of_death)

plt.show()
armed = police_killings.armed.value_counts()
plt.figure(figsize=(10,7))

sns.barplot(x = armed[:7].index, y = armed[:7].values)

plt.ylabel('Number of Weapon')

plt.xlabel('Weapon Types')

plt.title('Kill weapon',color = 'blue',fontsize=15)
above25 = ['above25' if i>25 else 'below25' for i in police_killings.age]

df = pd.DataFrame({'age': above25})

sns.countplot(x = df.age)

plt.ylabel('Number of Killed People')

plt.title('Age of killed people',color = 'blue',fontsize=15)
police_killings.head()
sns.countplot(x = police_killings.state)

plt.xticks(rotation = 90)

plt.show()
sns.countplot(police_killings.signs_of_mental_illness)

plt.show()
sns.countplot(police_killings.threat_level)

plt.show()