# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pandas as pd

import seaborn as sns

import numpy as np

import matplotlib.pyplot as plt

from collections import Counter

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# Read datas

median_house_hold_in_come = pd.read_csv('../input/MedianHouseholdIncome2015.csv', encoding="windows-1252")

percentage_people_below_poverty_level = pd.read_csv('../input/PercentagePeopleBelowPovertyLevel.csv', encoding="windows-1252")

percent_over_25_completed_highSchool = pd.read_csv('../input/PercentOver25CompletedHighSchool.csv', encoding="windows-1252")

share_race_city = pd.read_csv('../input/ShareRaceByCity.csv', encoding="windows-1252")

kill = pd.read_csv('../input/PoliceKillingsUS.csv', encoding="windows-1252")
#Poverty rate of each state

# replace '-' by 0.0 in the list percentage_people_below_poverty_level.poverty_rate. (or remove all 0.0s??)

percentage_people_below_poverty_level.poverty_rate.replace(['-'],0.0,inplace = True)

# change the type as float so that we can perform numeric operations on it.

percentage_people_below_poverty_level.poverty_rate = percentage_people_below_poverty_level.poverty_rate.astype(float)

# return unique items from the array and store them as a list.

area_list = list(percentage_people_below_poverty_level['Geographic Area'].unique())

# create a new empty array.

area_poverty_ratio = []

# running through each item in the list..

for i in area_list:

    # take in all the rows that have 'i' as the item in the column Geographic Area and store it in a new dataframe.

    x = percentage_people_below_poverty_level[percentage_people_below_poverty_level['Geographic Area']==i]

    # store the mean poverty rate as 'area_poverty_rate'

    area_poverty_rate = sum(x.poverty_rate)/len(x)

    # add the number into the list.

    area_poverty_ratio.append(area_poverty_rate)

# create new dataframe with all the area codes and their respective ratios.    

data = pd.DataFrame({'area_list': area_list,'area_poverty_ratio':area_poverty_ratio})

# sort the ratios in descending order and store their indexes into an array.

new_index = (data['area_poverty_ratio'].sort_values(ascending=False)).index.values

# reindex the Series and store it in a new dataframe.

sorted_data = data.reindex(new_index)

# Set the size for the plot.

plt.figure(figsize=(15,10))

# create barplot, x contains area list and y the poverty ratios.

ax= sns.barplot(x=sorted_data['area_list'], y=sorted_data['area_poverty_ratio'])

# set the x-lable orientations to be perpendicular to the axis. 

plt.xticks(rotation= 90)

# set the x-axis label

plt.xlabel('States')

# set the y-axis label

plt.ylabel('Poverty Rate')

# set the title for the plot

plt.title('Poverty Rate Given States')
# Most common 15 Name or Surname of killed people

# from the dataframe called 'separate' ignore rows in the array kill.name = 'TK TK'. Then split the names into first and second names.

separate = kill.name[kill.name != 'TK TK'].str.split() 

# separate the list kill.name in two.

a,b = zip(*separate) 

# combine each list into a single one and store it in a new one 'name_list'

name_list = a+b

# count the total number a name appears and store it as a dictionary.

name_count = Counter(name_list)      

# from the dictionary take the keys with the largest values and store them as a list.

most_common_names = name_count.most_common(15)

# unzip the list in two separate lists, names and their respective occurances.

x,y = zip(*most_common_names)

# turn x and y into lists.

x,y = list(x),list(y)

plt.figure(figsize=(15,10))

ax= sns.barplot(x=x, y=y)
#High school graduation rate of the population that is older than 25 in states

# replace all the rows with a '-' to 0.0 in the dataframe. 

percent_over_25_completed_highSchool.percent_completed_hs.replace(['-'],0.0,inplace = True)

# change all entries in the column 'percent_completed_hs' as float64 for numeric manipulation.

percent_over_25_completed_highSchool.percent_completed_hs = percent_over_25_completed_highSchool.percent_completed_hs.astype(float)

# store all unique entries in the column 'Geographic Area' in a list.

area_list = list(percent_over_25_completed_highSchool['Geographic Area'].unique())

# create a new empty array.

area_highschool = []

# iterate over all entries in the list of geographic areas.

for i in area_list:

    # ...... see above (same for loop)

    x = percent_over_25_completed_highSchool[percent_over_25_completed_highSchool['Geographic Area']==i]

    area_highschool_rate = sum(x.percent_completed_hs)/len(x)

    area_highschool.append(area_highschool_rate)

    

data = pd.DataFrame({'area_list': area_list,'area_highschool_ratio':area_highschool})

new_index = (data['area_highschool_ratio'].sort_values(ascending=True)).index.values

sorted_data2 = data.reindex(new_index)

plt.figure(figsize=(15,10))

ax= sns.barplot(x=sorted_data2['area_list'], y=sorted_data2['area_highschool_ratio'])

plt.xticks(rotation= 90)

plt.xlabel('States')

plt.ylabel('High School Graduate Rate')

plt.title("Percentage of Given State's Population Above 25 that Has Graduated High School")
# Percentage of state's population according to races that are black,white,native american, asian and hispanic

share_race_city.replace(['-'],0.0,inplace = True)

share_race_city.replace(['(X)'],0.0,inplace = True)

# choose selected columns to be float.

share_race_city.loc[:,['share_white','share_black','share_native_american','share_asian','share_hispanic']] = share_race_city.loc[:,['share_white','share_black','share_native_american','share_asian','share_hispanic']].astype(float)

# store all unique geographic areas in a list. 

area_list = list(share_race_city['Geographic area'].unique())

# create a few empty arrays.

share_white = []

share_black = []

share_native_american = []

share_asian = []

share_hispanic = []

# loop over entries in 'area_list'

for i in area_list:

    # take all rows in our df belonging to the area code corresp. to the ith element in our area list.

    x = share_race_city[share_race_city['Geographic area']==i]

    # take the average race for that area.

    share_white.append(sum(x.share_white)/len(x))

    share_black.append(sum(x.share_black) / len(x))

    share_native_american.append(sum(x.share_native_american) / len(x))

    share_asian.append(sum(x.share_asian) / len(x))

    share_hispanic.append(sum(x.share_hispanic) / len(x))

# superpose horizontal/vertical bar plots (AREA CODE ALONG Y-AXIS/PERCENTAGES ALONG X-AXIS... can exhange for a vertical plot.)

f,ax = plt.subplots(figsize = (15,9))

sns.barplot(x=area_list,y=share_white,color='green',alpha = 0.5,label='White' )

sns.barplot(x=area_list,y=share_black,color='blue',alpha = 0.7,label='African American')

sns.barplot(x=area_list,y=share_native_american,color='cyan',alpha = 0.6,label='Native American')

sns.barplot(x=area_list,y=share_asian,color='yellow',alpha = 0.6,label='Asian')

sns.barplot(x=area_list,y=share_hispanic,color='red',alpha = 0.6,label='Hispanic')



ax.legend(loc='lower right',frameon = True)     # legendlarin gorunurlugu

ax.set(xlabel='States', ylabel='Percentage of Races',title = "Percentage of State's Population According to Races ")
# high school graduation rate vs Poverty rate of each state

sorted_data['area_poverty_ratio'] = sorted_data['area_poverty_ratio']/max( sorted_data['area_poverty_ratio'])

sorted_data2['area_highschool_ratio'] = sorted_data2['area_highschool_ratio']/max( sorted_data2['area_highschool_ratio'])

data = pd.concat([sorted_data,sorted_data2['area_highschool_ratio']],axis=1)

data.sort_values('area_poverty_ratio',inplace=True)

f,ax1 = plt.subplots(figsize =(20,10))

sns.pointplot(x='area_list',y='area_poverty_ratio',data=data,color='lime',alpha=0.8,label = 'a')

sns.pointplot(x='area_list',y='area_highschool_ratio',data=data,color='red',alpha=0.8,label='b')

plt.text(40,0.6,'high school graduate ratio',color='red')

plt.text(40,0.55,'poverty ratio',color='lime')

plt.xlabel('States')

plt.ylabel('Values')

plt.title('High School Graduate  VS  Poverty Rate')

plt.grid()
# kill properties

# Manner of death

sns.countplot(kill.gender)

sns.countplot(kill.manner_of_death)
# kill weapon

armed = kill.armed.value_counts()

plt.figure(figsize=(10,7))

sns.barplot(x=armed[:7].index,y=armed[:7].values)

plt.ylabel('Number of Weapon')

plt.xlabel('Weapon Types')
# age of killed people

above25 =['above25' if i >= 25 else 'below25' for i in kill.age]

df = pd.DataFrame({'age':above25})

sns.countplot(x=df.age)

plt.ylabel('Number of Killed People')
# Race of killed people

sns.countplot(data=kill, x='race')
# Most dangerous cities

city = kill.city.value_counts()

plt.figure(figsize=(10,7))

sns.barplot(x=city[:12].index,y=city[:12].values)

plt.xticks(rotation=45)
# most dangerous states

state = kill.state.value_counts()

plt.figure(figsize=(10,7))

sns.barplot(x=state[:20].index,y=state[:20].values)
# Having mental ilness or not for killed people

sns.countplot(kill.signs_of_mental_illness)

plt.xlabel('Mental illness')

plt.ylabel('Number of Mental illness')
# Threat types

sns.countplot(kill.threat_level)

plt.xlabel('Threat Types')

# Flee types

sns.countplot(kill.flee)

plt.xlabel('Flee Types')
# Having body cameras or not for police

sns.countplot(kill.body_camera)

plt.xlabel('Having Body Cameras')


# Race rates according to states in kill data 

kill.race.dropna(inplace = True)

labels = kill.race.value_counts().index

colors = ['grey','blue','red','yellow','green','brown']

explode = [0,0,0,0,0,0]

sizes = kill.race.value_counts().values

plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%')

plt.title('Killed People According to Races')
# Kill numbers from states in kill data

sta = kill.state.value_counts().index[:10]

sns.barplot(x=sta,y = kill.state.value_counts().values[:10])