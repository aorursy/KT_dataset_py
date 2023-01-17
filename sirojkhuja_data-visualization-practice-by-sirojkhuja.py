import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

from collections import Counter

from warnings import filterwarnings
filterwarnings("ignore")

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
median_household_income_2015 = pd.read_csv("../input/fatal-police-shootings-in-the-us/MedianHouseholdIncome2015.csv",encoding = "windows-1252")
percent_over_25_completed_high_school = pd.read_csv("../input/fatal-police-shootings-in-the-us/PercentOver25CompletedHighSchool.csv",encoding = "windows-1252")
percentage_people_below_poverty_level = pd.read_csv("../input/fatal-police-shootings-in-the-us/PercentagePeopleBelowPovertyLevel.csv",encoding = "windows-1252")
police_killings = pd.read_csv("../input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv",encoding = "windows-1252")
share_race_by_city = pd.read_csv("../input/fatal-police-shootings-in-the-us/ShareRaceByCity.csv",encoding = "windows-1252")
percentage_people_below_poverty_level.poverty_rate.replace(["-"], 0.0, inplace = True)
# Poverty rate of each state
# percentage_people_below_poverty_level.poverty_rate = pd.to_numeric(percentage_people_below_poverty_level.poverty_rate)
percentage_people_below_poverty_level.poverty_rate = percentage_people_below_poverty_level.poverty_rate.astype(float)
percentage_people_below_poverty_level.info()
area_list = list(percentage_people_below_poverty_level["Geographic Area"].unique())
area_poverty_ratio = []

for area in area_list:
    x = percentage_people_below_poverty_level[percentage_people_below_poverty_level["Geographic Area"] == area]
    area_poverty_rate = sum(x.poverty_rate) / len(x)
    area_poverty_ratio.append(area_poverty_rate)

# sorting
data = pd.DataFrame({"area_list": area_list, "area_poverty_ratio": area_poverty_ratio}) 
sorted_index=(data["area_poverty_ratio"].sort_values(ascending=False)).index.values
sorted_data=data.reindex(sorted_index)

# visualization
plt.figure(figsize = (15,10))
sns.barplot(x = sorted_data["area_list"], y = sorted_data["area_poverty_ratio"])
plt.xticks(rotation = 90)
plt.xlabel("States")
plt.ylabel("Poverty Ratio")
plt.title("Poverty Rate For Given States")
plt.show()
police_killings.head()
firstnames_and_lastnames = police_killings.name[(police_killings.name != "TK TK") & (police_killings.name != "TK Tk")].str.split()
firstnames_and_lastnames
a,b = zip(*firstnames_and_lastnames) 
name_list = a + b

# following code was to see the content of name_list array
# name_list
name_count = Counter(name_list)
most_common_names = name_count.most_common(15)
x_data,y_data = zip(*most_common_names)
x_data,y_data = list(x_data),list(y_data)
 
plt.figure(figsize = (15,10))  
sns.barplot(x = x_data, y = y_data, palette = sns.cubehelix_palette(start = len(x),rot =-.4))
plt.xlabel("First or Last Names of Victims")
plt.ylabel("Frequency")
plt.title("Most Common 15 First or Last Names of Victims")
plt.show()
percent_over_25_completed_high_school.head()
percent_over_25_completed_high_school.info()
# percent_over_25_completed_high_school.percent_completed_hs.value_counts()
# High School graduation rate of the population that is older than 25 in states
percent_over_25_completed_high_school.percent_completed_hs.replace(["-"], 0.0, inplace = True)
percent_over_25_completed_high_school.percent_completed_hs = percent_over_25_completed_high_school.percent_completed_hs.astype(float)
area_list = list(percent_over_25_completed_high_school['Geographic Area'])
area_highschool_ratio = []
for area in area_list:
    x = percent_over_25_completed_high_school[percent_over_25_completed_high_school["Geographic Area"] == area]
    area_highschool_rate = sum(x.percent_completed_hs) / len(x)
    area_highschool_ratio.append(area_highschool_rate)

# sorting
data = pd.DataFrame({"area_list": area_list, "area_highschool_ratio": area_highschool_ratio})
sorted_index = (data['area_highschool_ratio'].sort_values(ascending = True)).index.values
sorted_data2 = data.reindex(sorted_index)

# draw a graph
plt.figure(figsize = (15,10))
sns.barplot(x = sorted_data2['area_list'], y = sorted_data2['area_highschool_ratio'])
plt.xticks(rotation = 90)
plt.xlabel("States")
plt.ylabel("High School Graduate Rate")
plt.title("Percentage of Given States' Population Above 25 Graduted High School")
plt.show()
share_race_by_city.head()
share_race_by_city.columns
# Percentage of state's population according to races that are black, white, native american, asian and hispanic
share_race_by_city.replace(["-"], 0.0, inplace = True)
share_race_by_city.replace(["(X)"], 0.0, inplace = True)
share_race_by_city.loc[:,['share_white','share_black','share_native_american','share_asian','share_hispanic']] = share_race_by_city.loc[:,['share_white','share_black','share_native_american','share_asian','share_hispanic']].astype(float)
area_list = list(share_race_by_city['Geographic area'].unique())
share_white = []
share_black = []
share_native_american = []
share_asian = []
share_hispanic = []

for area in area_list:
    x = share_race_by_city[share_race_by_city["Geographic area"] == area]
    share_white.append(sum(x.share_white)/len(x))
    share_black.append(sum(x.share_black)/len(x))
    share_native_american.append(sum(x.share_native_american)/len(x))
    share_asian.append(sum(x.share_asian)/len(x))
    share_hispanic.append(sum(x.share_hispanic)/len(x))

# draw a graph
f,ax = plt.subplots(figsize = (9,15))
sns.barplot(x = share_white, y = area_list, color = 'green', alpha = 0.5, label = 'White')
sns.barplot(x = share_black, y = area_list, color = 'blue', alpha = 0.7, label = 'Afro-American')
sns.barplot(x = share_native_american, y = area_list, color = 'cyan', alpha = 0.6, label = 'Native American')
sns.barplot(x = share_asian, y = area_list, color = 'yellow', alpha = 0.6, label = 'Asian')
sns.barplot(x = share_hispanic, y = area_list, color = 'red', alpha = 0.6, label = 'Hispanic')

ax.legend(loc = 'lower right', frameon = True)
ax.set(xlabel = "Percentage of Races", ylabel = "States", title = "Percentage of States' Population According to Races")

plt.show()
# High School graduation rate vs Poverty rate of each state
sorted_data['area_poverty_ratio'] = sorted_data['area_poverty_ratio']/max(sorted_data['area_poverty_ratio'])
sorted_data2['area_highschool_ratio'] = sorted_data2['area_highschool_ratio']/max(sorted_data2['area_highschool_ratio'])
data = pd.concat([sorted_data, sorted_data2['area_highschool_ratio']], axis = 1)
data.sort_values('area_poverty_ratio', inplace = True)

# draw a graph
f,ax1 = plt.subplots(figsize = (20,10))
sns.pointplot(x = 'area_list', y = 'area_poverty_ratio', data = data, color = 'lime', alpha = 0.8)
sns.pointplot(x = 'area_list', y = 'area_highschool_ratio', data = data, color = 'red', alpha = 0.8)
plt.text(40, 0.6, 'High School graduate ratio', color = 'red', fontsize = 17, style = 'italic')
plt.text(40, 0.55, 'Poverty Ratio', color = 'lime', fontsize = 18, style = 'italic')
plt.xlabel("States", fontsize = 15, color = "blue")
plt.ylabel("Values", fontsize = 15, color = "blue")
plt.title("High School Graduate VS Poverty Rate", fontsize = 20, color = "blue")
plt.grid()
plt.show()
