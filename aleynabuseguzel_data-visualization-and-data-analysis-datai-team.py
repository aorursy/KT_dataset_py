# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
import warnings
warnings.filterwarnings('ignore')
        
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
PercentOver25CompletedHighSchool = pd.read_csv('/kaggle/input/fatal-police-shootings-in-the-us/PercentOver25CompletedHighSchool.csv' , encoding = 'unicode_escape' )
PercentagePeopleBelowPovertyLevel = pd.read_csv('/kaggle/input/fatal-police-shootings-in-the-us/PercentagePeopleBelowPovertyLevel.csv' , encoding = 'unicode_escape')
MedianHouseholdIncome = pd.read_csv('/kaggle/input/fatal-police-shootings-in-the-us/MedianHouseholdIncome2015.csv' , encoding = 'unicode_escape')
ShareRaceByCity = pd.read_csv('/kaggle/input/fatal-police-shootings-in-the-us/ShareRaceByCity.csv' , encoding = 'unicode_escape')
PoliceKillingsUS = pd.read_csv('/kaggle/input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv' , encoding = 'unicode_escape')
PercentagePeopleBelowPovertyLevel.head()
PercentagePeopleBelowPovertyLevel.info()
PercentagePeopleBelowPovertyLevel.poverty_rate.value_counts()
#There were 1464 people with a poverty rate of 0. There are 201 data with unknown power rate.
# Poverty rate of each state
PercentagePeopleBelowPovertyLevel.poverty_rate.replace(['-'],0.0,inplace = True)
#Powerty rate equalized to 0 to ignore 201 meaningless data
PercentagePeopleBelowPovertyLevel.poverty_rate = PercentagePeopleBelowPovertyLevel.poverty_rate.astype(float)
#To be able to visualize defined as powerty rate object, it must be converted to float.
area_list = list(PercentagePeopleBelowPovertyLevel['Geographic Area'].unique())
#print(area_list)
area_poverty_ratio = []
for i in area_list:
    x = PercentagePeopleBelowPovertyLevel[PercentagePeopleBelowPovertyLevel['Geographic Area']==i]
    area_poverty_rate = sum(x.poverty_rate)/len(x)
    area_poverty_ratio.append(area_poverty_rate)
data = pd.DataFrame({'area_list': area_list,'area_poverty_ratio':area_poverty_ratio})
new_index = (data['area_poverty_ratio'].sort_values(ascending=False)).index.values
sorted_data = data.reindex(new_index)
#powerty rates in states are sorted in descending order


#Visualization
#Bar Plot to find the poorest state

plt.figure(figsize=(15,10))
sns.barplot(x=sorted_data['area_list'], y=sorted_data['area_poverty_ratio'])
plt.xticks(rotation= 45)
plt.xlabel('States')
plt.ylabel('Poverty Rate')
plt.title('Poverty Rate Given States')
plt.show()
PoliceKillingsUS.head()
PoliceKillingsUS.name.value_counts()
#There are 49 unknown names as TK. To find the 15 most common names, it is necessary to ignore the TK name.
separate = PoliceKillingsUS.name[PoliceKillingsUS.name != 'TK TK'].str.split()
a,b = zip(*separate)
name_list = a+b
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
PercentOver25CompletedHighSchool.head()
PercentOver25CompletedHighSchool.percent_completed_hs.value_counts()
#As in the previous data set, there are 197 unknown rates in this.
# High school graduation rate of the population that is older than 25 in states
PercentOver25CompletedHighSchool.percent_completed_hs.replace(['-'],0.0,inplace = True)
PercentOver25CompletedHighSchool.percent_completed_hs = PercentOver25CompletedHighSchool.percent_completed_hs.astype(float)
area_list = list(PercentOver25CompletedHighSchool['Geographic Area'].unique())
area_highschool = []
for i in area_list:
    x = PercentOver25CompletedHighSchool[PercentOver25CompletedHighSchool['Geographic Area']==i]
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
plt.show()
ShareRaceByCity.head()
