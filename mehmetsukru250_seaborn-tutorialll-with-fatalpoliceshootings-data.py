# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from collections import Counter

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
poverty=pd.read_csv("../input/PercentagePeopleBelowPovertyLevel.csv", encoding="windows-1252")

median = pd.read_csv('../input/MedianHouseholdIncome2015.csv', encoding="windows-1252")

completed= pd.read_csv('../input/PercentOver25CompletedHighSchool.csv', encoding="windows-1252")

share = pd.read_csv('../input/ShareRaceByCity.csv', encoding="windows-1252")

kill = pd.read_csv('../input/PoliceKillingsUS.csv', encoding="windows-1252")
poverty.head()
median.head()
completed.head()
share.head()
kill.head()
poverty.info()
poverty["Geographic Area"].unique()
poverty.poverty_rate.value_counts()
poverty.poverty_rate.replace(["-"],0.0,inplace=True)

poverty.poverty_rate=poverty.poverty_rate.astype(float)

area_list = list(poverty['Geographic Area'].unique())

area_poverty_ratio=[]

for i in area_list:

    x=poverty[poverty["Geographic Area"]==i]

    area_poverty_ratio.append(sum(x.poverty_rate)/len(x))

data=pd.DataFrame({'area_list':area_list,"area_poverty_ratio":area_poverty_ratio})

new_index = (data['area_poverty_ratio'].sort_values(ascending=False)).index.values

sorted_data=data.reindex(new_index)   

plt.figure(figsize=(15,10))

sns.barplot(x=sorted_data["area_list"],y=sorted_data["area_poverty_ratio"])

plt.xticks(rotation=45)

plt.xlabel("states")

plt.ylabel("Poverty rate")

plt.title("poverty rate for states")

plt.show()
#kill.name.value_counts()

separate=kill.name[kill.name!="TK TK"].str.split()

a,b=zip(*separate)#unzip

name_list=a+b

name_count = Counter(name_list)

common_names=name_count.most_common(15)

x,y=zip(*common_names)

x,y=list(x),list(y)

plt.figure(figsize=(15,10))

sns.barplot(x,y,palette=sns.cubehelix_palette(len(x)))

plt.xticks(rotation=-45)

plt.xlabel('Name or Surname of killed people')

plt.ylabel('Frequency')

plt.title('Most common 15 Name or Surname of killed people')

plt.show()

# High school graduation rate of the population that is older than 25 in states

completed.percent_completed_hs.replace(['-'],0.0,inplace = True)

completed.percent_completed_hs = completed.percent_completed_hs.astype(float)

area_list = list(completed['Geographic Area'].unique())

area_highschool = []

for i in area_list:

    x = completed[completed['Geographic Area']==i]

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
share.replace(['-'],0.0,inplace = True)

share.replace(['(X)'],0.0,inplace = True)

share.loc[:,['share_white','share_black','share_native_american','share_asian','share_hispanic']] = share.loc[:,['share_white','share_black','share_native_american','share_asian','share_hispanic']].astype(float)

area_list = list(share['Geographic area'].unique())

share_white = []

share_black = []

share_native_american = []

share_asian = []

share_hispanic = []

for i in area_list:

    x = share[share['Geographic area']==i]

    share_white.append(sum(x.share_white)/len(x))

    share_black.append(sum(x.share_black) / len(x))

    share_native_american.append(sum(x.share_native_american) / len(x))

    share_asian.append(sum(x.share_asian) / len(x))

    share_hispanic.append(sum(x.share_hispanic) / len(x))



# visualization

f,ax = plt.subplots(figsize = (9,15))

sns.barplot(x=share_white,y=area_list,color='green',alpha = 0.5,label='White' )

sns.barplot(x=share_black,y=area_list,color='blue',alpha = 0.7,label='African American')

sns.barplot(x=share_native_american,y=area_list,color='cyan',alpha = 0.6,label='Native American')

sns.barplot(x=share_asian,y=area_list,color='yellow',alpha = 0.6,label='Asian')

sns.barplot(x=share_hispanic,y=area_list,color='red',alpha = 0.6,label='Hispanic')



ax.legend(loc='lower right',frameon = True)     # legendlarin gorunurlugu

ax.set(xlabel='Percentage of Races', ylabel='States',title = "Percentage of State's Population According to Races ")