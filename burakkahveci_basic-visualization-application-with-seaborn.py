# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

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
median_house_hold_in_come.head()
#the glance at poverty rates

percentage_people_below_poverty_level.head()
percentage_people_below_poverty_level.info()
percentage_people_below_poverty_level.poverty_rate.value_counts()

#There are 201 data named "-". We must clean in data that is meaningless.
#Cleaning Data 

percentage_people_below_poverty_level.poverty_rate.replace(['-'],0.0,inplace=True)

#And after we should change from the object to the float. 

percentage_people_below_poverty_level.poverty_rate=percentage_people_below_poverty_level.poverty_rate.astype(float)
percentage_people_below_poverty_level['Geographic Area'].unique()

area_list = list(percentage_people_below_poverty_level['Geographic Area'].unique())
#Separation to States as per rates large to small with unique() method.

areapoveryratio=[]

for i in area_list:

    x=percentage_people_below_poverty_level[percentage_people_below_poverty_level['Geographic Area']==i]

    area_poverty_rate=sum(x.poverty_rate)/len(x)

    areapoveryratio.append(area_poverty_rate)

#We obtained poverty averages in all states. => area_poverty_rate
#creategraphicdata and indexing process



datagraphic=pd.DataFrame({'area_list':area_list,'areapoveryratio':areapoveryratio})

newindex=(datagraphic['areapoveryratio'].sort_values(ascending=False)).index.values

sorteddata=datagraphic.reindex(newindex)
#Visualization



plt.figure(figsize=(15,10)) #graphicsize

sns.barplot(x=sorteddata['area_list'],y=sorteddata['areapoveryratio'],palette="rocket")

plt.xticks(rotation=45)

plt.xlabel('States')

plt.ylabel('Poverty Rate')

plt.title('Poverty Rate Given States')

plt.show()
#Visualization of races separately with BarPlot

share_race_city.head()
share_race_city.info()
#cleaning data

share_race_city.replace(['-'],0.0,inplace=True)

share_race_city.replace(['(X)'],0.0,inplace=True)

#we must change from object to float races data.

share_race_city.loc[:,['share_white','share_black','share_native_american','share_asian','share_hispanic']] = share_race_city.loc[:,['share_white','share_black','share_native_american','share_asian','share_hispanic']].astype(float)
area_list = list(share_race_city['Geographic area'].unique())
#We obtained races averages in all states.

share_white = []

share_black = []

share_native_american = []

share_asian = []

share_hispanic = []

for i in area_list:

    x=share_race_city[share_race_city['Geographic area']==i]

    share_white.append(sum(x.share_white)/len(x))

    share_black.append(sum(x.share_black) / len(x))

    share_native_american.append(sum(x.share_native_american) / len(x))

    share_asian.append(sum(x.share_asian) / len(x))

    share_hispanic.append(sum(x.share_hispanic) / len(x))
plt.figure(figsize=(15,10)) #graphicsize

sns.barplot(x=area_list,y=share_white)

plt.xticks(rotation=45)

plt.xlabel('States')

plt.ylabel('Whites')

plt.title('Whites in States')

plt.show()
plt.figure(figsize=(15,10)) #graphicsize

sns.barplot(x=area_list,y=share_black,palette = sns.cubehelix_palette(len(share_white)))

plt.xticks(rotation=45)

plt.xlabel('States')

plt.ylabel('African American')

plt.title('African American in States')

plt.show()
plt.figure(figsize=(15,10)) #graphicsize

sns.barplot(x=area_list,y=share_native_american)

plt.xticks(rotation=45)

plt.xlabel('States')

plt.ylabel('Native American')

plt.title('Native American in States')

plt.show()
plt.figure(figsize=(15,10)) #graphicsize

sns.barplot(x=area_list,y=share_asian)

plt.xticks(rotation=45)

plt.xlabel('States')

plt.ylabel('Asian')

plt.title('Asian in States')

plt.show()
plt.figure(figsize=(15,10)) #graphicsize

sns.barplot(x=area_list,y=share_hispanic)

plt.xticks(rotation=45)

plt.xlabel('States')

plt.ylabel('Hispanic')

plt.title('Hispanic in States')

plt.show()
dataraces=pd.DataFrame(({'area_list':area_list,'share_white':share_white,'share_hispanic':share_hispanic,'share_asian':

                        share_asian,'share_native_american':share_native_american,'share_black':share_black}))
kill.head()
#To Separate as per mental states

mentalillnes=['mentalillness'if i ==True else'normal'for i in kill.signs_of_mental_illness]

df = pd.DataFrame({'mental_states':mentalillnes})
#Visualization

sns.countplot(x=mentalillnes)

plt.ylabel('Number of Killed People')

plt.title('Mental States of killed people',color = 'black',fontsize=15)

plt.show()
sns.swarmplot(x="gender",y="age",hue="signs_of_mental_illness",data=kill)

plt.show()
sns.boxplot(x="gender", y="age", hue="signs_of_mental_illness", data=kill, palette="PRGn")

plt.show()