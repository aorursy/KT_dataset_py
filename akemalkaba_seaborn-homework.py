# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
%matplotlib inline

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
median_household_income = pd.read_csv('../input/MedianHouseholdIncome2015.csv',encoding="windows-1252")
percentage_people_below_poverty =  pd.read_csv('../input/PercentagePeopleBelowPovertyLevel.csv',encoding="windows-1252")
percentage_Over25_completed_highschool = pd.read_csv('../input/PercentOver25CompletedHighSchool.csv',encoding="windows-1252")
police_killings = pd.read_csv('../input/PoliceKillingsUS.csv',encoding="windows-1252")
share_race_city = pd.read_csv('../input/ShareRaceByCity.csv',encoding="windows-1252")
percentage_people_below_poverty.head()
percentage_people_below_poverty.info()
percentage_people_below_poverty.poverty_rate.value_counts()
percentage_people_below_poverty['Geographic Area'].unique()
percentage_people_below_poverty.poverty_rate.replace(['-'],0,inplace =True)
percentage_people_below_poverty.poverty_rate = percentage_people_below_poverty.poverty_rate.astype(float)
percentage_people_below_poverty.poverty_rate.value_counts()
area_list = list(percentage_people_below_poverty['Geographic Area'].unique())
area_poverty_ratio = []
for i in area_list:
    x=percentage_people_below_poverty[percentage_people_below_poverty['Geographic Area'] == i]
    area_poverty_rate = sum(x.poverty_rate)/len(x)
    area_poverty_ratio.append(area_poverty_rate)
data = pd.DataFrame({'area_list':area_list,'area_poverty_ratio':area_poverty_ratio})
new_index = (data['area_poverty_ratio'].sort_values(ascending=False)).index.values
sorted_data = data.reindex(new_index)

plt.figure(figsize=(15,10))
sns.barplot(x=sorted_data['area_list'],y=sorted_data['area_poverty_ratio'])
plt.xticks(rotation =45)
plt.xlabel('States')
plt.ylabel('Poverty Rate')
plt.title('Poverty Rate Given States')

police_killings.head()
seperate = police_killings.name[police_killings.name != 'TK TK'].str.split()
a,b = zip(*seperate)
name_list = a+b
name_count = Counter(name_list)
most_common_names =name_count.most_common(15)
x,y = zip (*most_common_names)
x,y = list(x),list(y)

plt.figure(figsize=(15,10))
ax =sns.barplot(x=x,y=y,palette = sns.cubehelix_palette(len(x)))
plt.xlabel('Name or Surname of Killed')
plt.ylabel('Frequency')
plt.title('Most Common 15 Names or Surnames of Killed People')
percentage_Over25_completed_highschool.head()
percentage_Over25_completed_highschool.head()
percentage_Over25_completed_highschool.percent_completed_hs.replace(['-'],0.0,inplace = True)
percentage_Over25_completed_highschool.percent_completed_hs =percentage_Over25_completed_highschool.percent_completed_hs.astype(float)
area_list = list(percentage_Over25_completed_highschool['Geographic Area'].unique())
area_highschool = []
for i in area_list:
    x=percentage_Over25_completed_highschool[percentage_Over25_completed_highschool['Geographic Area']==i]
    area_highschool_rate = sum(x.percent_completed_hs)/len(x)
    area_highschool.append(area_highschool_rate)

data = pd.DataFrame({'area_list':area_list,'area_highschool_ratio':area_highschool})
new_index = (data['area_highschool_ratio'].sort_values(ascending=True)).index.values
sorted_data2 =data.reindex(new_index) 

plt.figure(figsize=(15,10))
sns.barplot(x=sorted_data2['area_list'],y=sorted_data2['area_highschool_ratio'])
plt.xticks(rotation=90)
plt.xlabel('States')
plt.ylabel('High School Graduate Rate')
plt.title('Perentage of Given State Above 25 grad ghigh school')



share_race_by_city.head()
share_race_by_city.info()
share_race_city.replace(['-'],0,inplace=True)
share_race_city.replace(['X'],0,inplace=True)
share_race_city.replace(['(X)'],0.0,inplace=True)
share_race_city.loc[:,['share_white','share_black','share_native_american','share_asian','share_hispanic']] = share_race_city.loc[:,['share_white','share_black','share_native_american','share_asian','share_hispanic']].astype(float)
area_list = list(share_race_city['Geographic area'].unique())
share_white = []
share_black = []
share_native_american =[]
share_asian = []
share_hispanic =[]

for i in area_list:
    x = share_race_city[share_race_city['Geographic area']==i]
    share_white.append(sum(x.share_white)/len(x))
    #share_black.append =(sum(x.share_black)/len(x))
    #share_native_american.append =(sum(x.share_native_american)/len(x))
    #share_asian.append = (sum(x.share_asian)/len(x))
    #share_hispanic.append =(sum(x.share_hispanic)/len(x))



