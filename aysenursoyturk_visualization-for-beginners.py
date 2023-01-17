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
poverty_level = pd.read_csv('../input/PercentagePeopleBelowPovertyLevel.csv',encoding="windows-1252")
over_25_completed_highSchool = pd.read_csv('../input/PercentOver25CompletedHighSchool.csv',encoding="windows-1252")
kill = pd.read_csv('../input/PoliceKillingsUS.csv',encoding="windows-1252")
poverty_level.head()
poverty_level.info()
poverty_level.poverty_rate.value_counts()
poverty_level.poverty_rate.replace(['-'],0.0,inplace=True)
poverty_level.poverty_rate = poverty_level.poverty_rate.astype(float)
poverty_level.info()
area_list = list(poverty_level['Geographic Area'].unique())
area_poverty_ratio = []
for i in area_list:
    x = poverty_level[poverty_level['Geographic Area']==i]
    area_poverty_rate = sum(x.poverty_rate)/len(x)
    area_poverty_ratio.append(area_poverty_rate)
data = pd.DataFrame({'area_list': area_list,'area_poverty_ratio':area_poverty_ratio})
new_index = (data['area_poverty_ratio'].sort_values(ascending=False)).index.values
sorted_data = data.reindex(new_index)


plt.figure(figsize=(15,10))
sns.barplot(x=sorted_data['area_list'], y=sorted_data['area_poverty_ratio'],palette = sns.cubehelix_palette(len(x)))
plt.xticks(rotation= 37.5)
plt.xlabel('States')
plt.ylabel('Poverty Rate')
plt.title('Poverty Rate Given States')
kill.head()
kill.info()
kill.name.value_counts()
separate = kill.name[kill.name != 'TK TK'].str.split()
#print(separate)
a,b = zip(*separate)                    
name_list = a+b
#print(name_list)
name_count = Counter(name_list)
#print(name_count)
most_common_names = name_count.most_common(15)
x,y = zip(*most_common_names)
x,y = list(x),list(y)
#visualization
plt.figure(figsize=(15,10))
ax= sns.barplot(x=x, y=y)
plt.xlabel('Name or Surname of killed people')
plt.ylabel('Frequency')
plt.title('Most common 15 Name or Surname of killed people')
over_25_completed_highSchool.head()
over_25_completed_highSchool.info()
over_25_completed_highSchool.percent_completed_hs.value_counts()
over_25_completed_highSchool.percent_completed_hs.replace(['-'],0.0,inplace=True)
over_25_completed_highSchool.percent_completed_hs = over_25_completed_highSchool.percent_completed_hs.astype(float)
over_25_completed_highSchool.info()
area_list = list(over_25_completed_highSchool['Geographic Area'].unique())
area_highschool = []
for i in area_list:
    x = over_25_completed_highSchool[over_25_completed_highSchool['Geographic Area'] == i]
    area_highschool_rate = sum(x.percent_completed_hs)/len(x)
    area_highschool.append(area_highschool_rate)
data = pd.DataFrame({'area_list': area_list,'area_highschool_ratio':area_highschool})
new_index = (data['area_highschool_ratio'].sort_values(ascending=True)).index.values
sorted_data2 = data.reindex(new_index)
plt.figure(figsize=(15,10))
sns.barplot(x=sorted_data2['area_list'],y=sorted_data2['area_highschool_ratio'])
plt.xticks(rotation=37.5)
plt.xlabel('States')
plt.ylabel('High School Graduate Rate')
plt.title("Percentage of Given State's Population Above 25 that Has Graduated High School")
sorted_data.head()
sorted_data2.head()
sorted_data['area_poverty_ratio'] = sorted_data['area_poverty_ratio']/max( sorted_data['area_poverty_ratio'])
sorted_data2['area_highschool_ratio'] = sorted_data2['area_highschool_ratio']/max( sorted_data2['area_highschool_ratio'])
data = pd.concat([sorted_data,sorted_data2['area_highschool_ratio']],axis=1)
data.head()
data.sort_values('area_poverty_ratio',inplace=True)
data.head()
f,ax1 = plt.subplots(figsize =(20,10))
sns.pointplot(x='area_list',y='area_poverty_ratio',data=data,color='lime',alpha=0.8)
sns.pointplot(x='area_list',y='area_highschool_ratio',data=data,color='red',alpha=0.8)
plt.text(10,0.7,'high school graduate ratio',color='red',fontsize = 17,style = 'italic')
plt.text(10,0.65,'poverty ratio',color='lime',fontsize = 18,style = 'italic')
plt.xlabel('States',fontsize = 15,color='blue')
plt.ylabel('Values',fontsize = 15,color='blue')
plt.title('High School Graduate  VS  Poverty Rate',fontsize = 20,color='blue')
plt.grid()
data.head()
a = sns.jointplot(data.area_poverty_ratio, data.area_highschool_ratio, kind="kde", size=7)
plt.savefig('graph.png')
plt.show()
