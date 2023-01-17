# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from collections import Counter

%matplotlib inline



import warnings

warnings.filterwarnings('ignore') 



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
median_house_hold_in_come = pd.read_csv('/kaggle/input/fatal-police-shootings-in-the-us/MedianHouseholdIncome2015.csv',encoding="windows-1252")

percentage_people_below_poverty_level = pd.read_csv('/kaggle/input/fatal-police-shootings-in-the-us/PercentagePeopleBelowPovertyLevel.csv',encoding="windows-1252")

percent_over_25_completed_highSchool = pd.read_csv('/kaggle/input/fatal-police-shootings-in-the-us/PercentOver25CompletedHighSchool.csv',encoding="windows-1252")

share_race_city = pd.read_csv('/kaggle/input/fatal-police-shootings-in-the-us/ShareRaceByCity.csv',encoding="windows-1252")

kill = pd.read_csv('/kaggle/input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv',encoding="windows-1252")
percentage_people_below_poverty_level.head()
percentage_people_below_poverty_level.info()
percentage_people_below_poverty_level.poverty_rate.value_counts()
percentage_people_below_poverty_level.poverty_rate.replace(["-"],0.0, inplace=True)

percentage_people_below_poverty_level.poverty_rate = percentage_people_below_poverty_level.poverty_rate.astype(float)

area_list = list(percentage_people_below_poverty_level['Geographic Area'].unique())



area_proverty_ratio = []

for i in area_list:

    x = percentage_people_below_poverty_level[percentage_people_below_poverty_level['Geographic Area']== i]

    area_proverty_rate = sum(x.poverty_rate)/len(x)

    area_proverty_ratio.append(area_proverty_rate)



data = pd.DataFrame({"area_list": area_list, "area_proverty_ratio": area_proverty_ratio})

new_index = (data['area_proverty_ratio'].sort_values(ascending=False)).index.values

sorted_data = data.reindex(new_index)



plt.figure(figsize=(15,10))

sns.barplot(x=sorted_data['area_list'],y=sorted_data['area_proverty_ratio'])

plt.xticks(rotation=90) #apsis font transform

plt.xlabel('States')

plt.ylabel('Poverty Rate')

plt.title('Poverty Rate Given States')

plt.show()
kill.head()
kill.name.value_counts()
separate = kill.name[kill.name != 'TK TK'].str.split()

a,b = zip(*separate)

name_list = a + b

name_count = Counter(name_list)

most_common_names = name_count.most_common(15)

x,y = zip(*most_common_names)

x,y = list(x),list(y)



plt.figure(figsize=(15,10))

ax= sns.barplot(x=x, y=y,palette = sns.cubehelix_palette(len(x)))

plt.xlabel('Name or Surname of killed people')

plt.ylabel('Frequency')

plt.title('Most common 15 Name or Surname of killed people')

plt.show()
percent_over_25_completed_highSchool.head()
percent_over_25_completed_highSchool.percent_completed_hs.value_counts()
percent_over_25_completed_highSchool.percent_completed_hs.replace(["-"],0.0,inplace=True)

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

# visualization

plt.figure(figsize=(15,10))

sns.barplot(x=sorted_data2['area_list'], y=sorted_data2['area_highschool_ratio'])

plt.xticks(rotation= 90)

plt.xlabel('States')

plt.ylabel('High School Graduate Rate')

plt.title("Percentage of Given State's Population Above 25 that Has Graduated High School")
share_race_city.head()
share_race_city.info()
share_race_city.replace(["-"],0.0,inplace=True)

share_race_city.replace(["(X)"],0.0,inplace=True)

share_race_city.loc[:,['share_white','share_black','share_native_american','share_asian','share_hispanic']] = share_race_city.loc[:,['share_white','share_black','share_native_american','share_asian','share_hispanic']].astype(float)

area_list = list(share_race_city["Geographic area"].unique())



share_white = []

share_black = []

share_native_american = []

share_asian = []

share_hispanic = []



for i in area_list:

    x = share_race_city[share_race_city["Geographic area"] == i]

    share_white.append(sum(x.share_white)/len(x))

    share_black.append(sum(x.share_black)/len(x))

    share_native_american.append(sum(x.share_native_american)/len(x))

    share_asian.append(sum(x.share_asian)/len(x))

    share_hispanic.append(sum(x.share_asian)/len(x))



f, ax = plt.subplots(figsize = (9,15))

sns.barplot(x=share_white, y=area_list, alpha=0.5,color="red",label="White")

sns.barplot(x=share_black, y=area_list, alpha=0.5,color="blue",label="Black")

sns.barplot(x=share_native_american, y=area_list, alpha=0.5, color="cyan",label="Native American")

sns.barplot(x=share_asian, y=area_list, alpha=0.5, color="green", label="Asian")

sns.barplot(x=share_hispanic, y=area_list, alpha=0.5, color="orange", label="Hispanic")

ax.legend(loc="lower right",frameon=True)

ax.set(xlabel="Percentage of Races", ylabel="States",title="Percentage of State's Population According to Races")
# high school graduation rate vs Poverty rate of each state



sorted_data2['area_highschool_ratio'] = sorted_data2['area_highschool_ratio']/max(sorted_data2['area_highschool_ratio'])

sorted_data['area_proverty_ratio'] = sorted_data['area_proverty_ratio']/max(sorted_data['area_proverty_ratio'])



data0 = pd.concat([sorted_data,sorted_data2['area_highschool_ratio']],axis=1)

data0.sort_values('area_proverty_ratio',inplace=True)



f1, ax = plt.subplots(figsize = (20,10))

sns.pointplot(x='area_list',y='area_proverty_ratio',data=data0,color='lime',alpha=0.7)

sns.pointplot(x='area_list',y='area_highschool_ratio',data=data0,color='red',alpha=0.7)

plt.text(40,0.6,"high school ratio",fontsize=15,color='red',style='italic')

plt.text(40,0.55,"proverty ratio",fontsize=15,color='lime',style='italic')

plt.xlabel("States",fontsize=15)

plt.ylabel("Values",fontsize=15)

plt.title("Proverty Ratio VS High School Ratio",fontsize=20,color='blue',style='italic')

plt.grid()