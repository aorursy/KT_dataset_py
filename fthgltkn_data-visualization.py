# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
median_house_hold_in_come = pd.read_csv('../input/MedianHouseholdIncome2015.csv', encoding="windows-1252")

percentage_people_below_poverty_level = pd.read_csv('../input/PercentagePeopleBelowPovertyLevel.csv', encoding="windows-1252")

percent_over_25_completed_highSchool = pd.read_csv('../input/PercentOver25CompletedHighSchool.csv', encoding="windows-1252")

share_race_city = pd.read_csv('../input/ShareRaceByCity.csv', encoding="windows-1252")

kill = pd.read_csv('../input/PoliceKillingsUS.csv', encoding="windows-1252")
percentage_people_below_poverty_level.columns
#percentage_people_below_poverty_level.poverty_rate.value_counts()
percentage_people_below_poverty_level.info()
percentage_people_below_poverty_level.poverty_rate.replace(['-'],0.0,inplace=True)

percentage_people_below_poverty_level.poverty_rate=percentage_people_below_poverty_level.poverty_rate.astype(float)

area_list=list(percentage_people_below_poverty_level['Geographic Area'])

poverty_area_ratio=[]

for i in area_list:

    x=percentage_people_below_poverty_level[percentage_people_below_poverty_level['Geographic Area']==i]

    poverty_area=(sum(x.poverty_rate)/len(x))

    poverty_area_ratio.append(poverty_area)

    

data=pd.DataFrame({'area_list':area_list,'poverty_area_ratio':poverty_area_ratio})

new_index=(data['poverty_area_ratio'].sort_values(ascending=False)).index.values

sorted_data1=data.reindex(new_index)

#visualization



plt.figure(figsize=(15,15))

sns.barplot(x=sorted_data1['area_list'],y=sorted_data1['poverty_area_ratio'])

plt.xtick(rotation=45)

plt.xlabel('area_list')

plt.ylabel('poverty_area_ratio')

plt.title('poverty_area_ratio')

plt.show()











kill.head()
#kill.name.value_counts()
#kill.info()
seperate=kill.name[kill.name!='TK TK'].str.split()

a,b=zip(*seperate)

name_list=a+b

name_count = Counter(name_list)

most_common_name=name_count.most_common(15)

x,y=zip(*most_common_name)

x,y=list(x),list(y)

plt.figure(figsize=(15/10))

sns.barplot(x=x,y=y,color='blue')

plt.xlabel('killed_name')

plt.ylabel('frequency')

plt.show()

percent_over_25_completed_highSchool.head()
#percent_over_25_completed_highSchool.percent_completed_hs.value_counts()
percent_over_25_completed_highSchool.info()
percent_over_25_completed_highSchool.percent_completed_hs.replace(['-'],0.0,inplace=True)

percent_over_25_completed_highSchool.percent_completed_hs=percent_over_25_completed_highSchool.percent_completed_hs.astype(float)

area_list=list(percent_over_25_completed_highSchool['Geographic Area'])

percent_completed_hs_ratio=[]

for i in area_list:

    x=percent_over_25_completed_highSchool[percent_over_25_completed_highSchool['Geographic Area']==i]

    completed_highSchool_rate=sum(x.percent_completed_hs)/len(x)

    percent_completed_hs_ratio.append(completed_highSchool_rate)

    

data=pd.DataFrame({'area_list':area_list,'percent_completed_hs_ratio':percent_completed_hs_ratio})

new_index=(data['percent_completed_hs_ratio'].sort_values(ascending=True)).index.values

sorted_data2=data.reindex(new_index)



plt.figure(figsize=(15,15))

sns.barplot(x=sorted_data2['area_list'],y=sorted_data2['percent_completed_hs_ratio']) 

plt.xkabel('state')

plt.ylabel('highschool graduate rate')

plt.xticks(rotation=45)

plt.title('percentage over 25 copmleted graduated school rate')

plt.show()
share_race_city.head()

share_race_city.info()
share_race_city.replace(['-'],0.0,inplace=True)

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





f,ax = plt.subplots(figsize = (9,15))

sns.barplot(x=share_white,y=area_list,color='green',alpha = 0.5,label='White' )

sns.barplot(x=share_black,y=area_list,color='blue',alpha = 0.7,label='African American')

sns.barplot(x=share_native_american,y=area_list,color='cyan',alpha = 0.6,label='Native American')

sns.barplot(x=share_asian,y=area_list,color='yellow',alpha = 0.6,label='Asian')

sns.barplot(x=share_hispanic,y=area_list,color='red',alpha = 0.6,label='Hispanic')



ax.legend(loc='lower right',frameon = True)     

ax.set(xlabel='Percentage of Races', ylabel='States',title = "Percentage of State's Population According to Races ")

















sorted_data1['poverty_area_ratio']=sorted_data1['poverty_area_ratio']/max(sorted_data1['poverty_area_ratio'])

sorted_data2['percent_completed_hs_ratio']=sorted_data2['percent_completed_hs_ratio']/max(sorted_data2['percent_completed_hs_ratio'])

data=pd.concat([sorted_data1,sorted_data2['percent_completed_hs_ratio']],axis=1)

data.sort_values('poverty_area_ratio',inplace=True)



f,ax1 = plt.subplots(figsize =(20,10))

sns.pointplot(x='area_list',y='poverty_area_ratio',data=data,color='red')

sns.pointplot(x='area_list',y='percent_completed_hs_ratio',data=data,color='blue')

plt.text(40,0.6,'high school graduate ratio',color='red',fontsize = 17,style = 'italic')

plt.text(40,0.55,'poverty ratio',color='lime',fontsize = 18,style = 'italic')

plt.xlabel('States',fontsize = 15,color='blue')

plt.ylabel('Values',fontsize = 15,color='blue')

plt.title('High School Graduate  VS  Poverty Rate',fontsize = 20,color='blue')

plt.grid()

data.head()
g = sns.jointplot(data.poverty_area_ratio, data.percent_completed_hs_ratio, kind="kde", size=7)

plt.savefig('graph.png')

plt.show()
data.head()
sns.lmplot(x='poverty_area_ratio',y='percent_completed_hs_ratio',data=data)

plt.show()
sns.kdeplot(data.poverty_area_ratio,data.percent_completed_hs_ratio,shade=True,cut=5)

plt.show()
sns.violinplot(data=data,inner='points')

plt.show()
kill.head()
sns.boxplot(x='gender',y='age',hue='manner_of_death',data=kill)

plt.show()
sns.swarmplot(x='gender',y='age',hue='manner_of_death',data=kill)

plt.figure(figsize=(10,10))

plt.show()
kill.race.value_counts()
kill.race.dropna(inplace=True)

labels=kill.race.value_counts.index

colors=['dark','blue','yellow','grey','green','pink']

explode=[0,0,0,0,0,0]

sizes=kill.race.value_counts().values

plt.pie(sizes,explode=explode,colors=colors,labels=labels)

plt.title('pie chart kill race',color='blue')