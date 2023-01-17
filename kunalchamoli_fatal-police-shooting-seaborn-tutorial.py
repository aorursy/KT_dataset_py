# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



from collections import Counter

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from subprocess import check_output     #to handle UTF-8 decode error

print(check_output(["ls", "../input"]).decode("utf8"))
median_household_income = pd.read_csv("../input/MedianHouseholdIncome2015.csv", encoding="windows-1252")

percent_people_below_poverty = pd.read_csv("../input/PercentagePeopleBelowPovertyLevel.csv", encoding="windows-1252")

percent_over_25 = pd.read_csv("../input/PercentOver25CompletedHighSchool.csv", encoding="windows-1252")

police_killing = pd.read_csv("../input/PoliceKillingsUS.csv", encoding="windows-1252")

Share_raceby_city = pd.read_csv("../input/ShareRaceByCity.csv", encoding="windows-1252")
percent_people_below_poverty.head(10)
area_list = list(percent_people_below_poverty['Geographic Area'].unique())
percent_people_below_poverty.isnull().sum(axis = 0)
percent_people_below_poverty.poverty_rate.replace(['-'],0.0,inplace = True)

percent_people_below_poverty.poverty_rate = percent_people_below_poverty.poverty_rate.astype(float)
df = percent_people_below_poverty.groupby(['Geographic Area']).mean()

df
type(df)
new_index = (df['poverty_rate'].sort_values(ascending = False)).index.values

sorted_df = df.reindex(new_index)

sorted_df['Geographic Area'] = sorted_df.index

sorted_df
#visualization

plt.figure(figsize = (12, 10))

sns.barplot(x = sorted_df['Geographic Area'], y = sorted_df['poverty_rate'])

plt.xticks(rotation = 45)

plt.xlabel('States')

plt.ylabel('Average Poverty Rate')

plt.title('Average Poverty Rate of Each State')
kill = police_killing.copy()

kill.head(10)
kill.age.value_counts().head()
#most common 15 Name or surname of people killed

seperate = kill['name'].str.split()

a, b = zip(*seperate)

name_list = a+ b

name_count = Counter(name_list)

most_common_names = name_count.most_common(15)  

x,y = zip(*most_common_names)

x,y = list(x),list(y)
plt.figure(figsize=(15,10))

ax= sns.barplot(x=x, y=y)

plt.xlabel('Name or Surname of killed people')

plt.ylabel('Frequency')

plt.title('Most common 15 Name or Surname of killed people')
Share_raceby_city.head(10)
Share_raceby_city.replace(['-'],0.0,inplace = True)

Share_raceby_city.replace(['(X)'],0.0,inplace = True)

Share_raceby_city.loc[:,['share_white','share_black','share_native_american','share_asian','share_hispanic']] =  Share_raceby_city.loc[:,['share_white','share_black','share_native_american',

                                                                                                                                          'share_asian','share_hispanic']].astype(float)
df2 = Share_raceby_city.groupby(['Geographic area']).mean()

type(df2)
df2['area_list'] = df2.index

df2.head(10)
f,ax = plt.subplots(figsize = (9,15))

sns.barplot(x=df2.share_white,y=df2.area_list, color='green',alpha = 0.5,label='White' )

sns.barplot(x=df2.share_black,y=df2.area_list,color='blue',alpha = 0.7,label='African American')

sns.barplot(x=df2.share_native_american,y=df2.area_list,color='cyan',alpha = 0.6,label='Native American')

sns.barplot(x=df2.share_asian,y=df2.area_list,color='yellow',alpha = 0.6,label='Asian')

sns.barplot(x=df2.share_hispanic,y=df2.area_list,color='red',alpha = 0.6,label='Hispanic')



ax.legend(loc='lower right',frameon = True)     

ax.set(xlabel='Percentage of Races', ylabel='States',title = "Percentage of State's Population According to Races")