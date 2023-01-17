# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # data visualization tool
import matplotlib.pyplot as plt 

from collections import Counter
%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Read csv_data with pandas 
percent_over_25_completed_high_school = pd.read_csv("../input/PercentOver25CompletedHighSchool.csv",encoding="windows-1252")
median_household_income_2015 = pd.read_csv("../input/MedianHouseholdIncome2015.csv", encoding="windows-1252")
share_raceby_city = pd.read_csv("../input/ShareRaceByCity.csv", encoding="windows-1252")
kill = pd.read_csv("../input/PoliceKillingsUS.csv", encoding="windows-1252")
percentage_people_below_poverty_level = pd.read_csv("../input/PercentagePeopleBelowPovertyLevel.csv", encoding="windows-1252")
percentage_people_below_poverty_level.head()
percentage_people_below_poverty_level.poverty_rate.value_counts()
percentage_people_below_poverty_level.info()
percentage_people_below_poverty_level.poverty_rate.replace(["-"],0.0,inplace = True)
percentage_people_below_poverty_level.poverty_rate = percentage_people_below_poverty_level.poverty_rate.astype(float)
geo_area = list(percentage_people_below_poverty_level["Geographic Area"].unique())
geo_area_ratio = []
for i in geo_area:
    x = percentage_people_below_poverty_level[percentage_people_below_poverty_level['Geographic Area']==i]
    #print(x)
    geo_area_poverty_rate = sum(x.poverty_rate) / len(x)
    geo_area_ratio.append(geo_area_poverty_rate)
df= pd.DataFrame({"geo_area" : geo_area, "geo_area_poverty_ratio" : geo_area_ratio})
new_index = (df['geo_area_poverty_ratio'].sort_values(ascending=False)).index.values
sorted_df = df.reindex(new_index)

#Visualization
plt.figure(figsize=(15,10))
sns.barplot(x = sorted_df["geo_area"], y = sorted_df["geo_area_poverty_ratio"])
plt.xticks(rotation = 45)
plt.ylabel("Poverty Rate")
plt.xlabel("Geographic Area")

# Read File
share_raceby_city = pd.read_csv("../input/ShareRaceByCity.csv", encoding="windows-1252")
share_raceby_city.head()
share_raceby_city.share_white.value_counts()
#share_raceby_city.share_black.value_counts()
#share_raceby_city.share_native_american.value_counts()
#share_raceby_city.share_asian.value_counts()
#share_raceby_city.share_hispanic.value_counts()
share_raceby_city.replace(['-'],0.0,inplace = True)
share_raceby_city.replace(['(X)'],0.0,inplace = True)

area_list = list(share_raceby_city['Geographic area'].unique())
share_raceby_city.loc[:,['share_white','share_black','share_native_american','share_asian','share_hispanic']] = share_raceby_city.loc[:,['share_white','share_black','share_native_american','share_asian','share_hispanic']].astype(float)
share_white = []
share_black = []
share_native_american = []
share_asian = []
share_hispanic = []

for i in area_list:
    x = share_raceby_city[share_raceby_city['Geographic area'] == i]
    share_white.append(sum(x.share_white)/len(x))
    share_black.append(sum(x.share_black)/len(x))
    share_native_american.append(sum(x.share_native_american) / len(x))
    share_asian.append(sum(x.share_asian) / len(x))
    share_hispanic.append(sum(x.share_hispanic) / len(x))
    
f ,ax = plt.subplots(figsize = (9,15))
sns.barplot(x=share_white, y=area_list , color='red', alpha = 0.5 , label = 'White')
sns.barplot(x=share_black,y=area_list,color='blue',alpha = 0.7,label='African American')
sns.barplot(x=share_native_american,y=area_list,color='cyan',alpha = 0.6,label='Native American')
sns.barplot(x=share_asian,y=area_list,color='yellow',alpha = 0.6,label='Asian')
sns.barplot(x=share_hispanic,y=area_list,color='green',alpha = 0.6,label='Hispanic')

ax.legend(loc='lower right',frameon = True)
ax.set(xlabel='Percentage of Races', ylabel='States',title = "Percentage of State's Population According to Races ")

percent_over_25_completed_high_school = pd.read_csv("../input/PercentOver25CompletedHighSchool.csv",encoding="windows-1252")
percentage_people_below_poverty_level = pd.read_csv("../input/PercentagePeopleBelowPovertyLevel.csv", encoding="windows-1252")

percent_over_25_completed_high_school.head()
#percent_over_25_completed_high_school.info()
percentage_people_below_poverty_level.head()
percent_over_25_completed_high_school.replace(['-'],0.0,inplace = True)
percent_over_25_completed_high_school.replace(['(X)'],0.0,inplace = True)
percentage_people_below_poverty_level.replace(('(X)'), 0.0,inplace = True)
percentage_people_below_poverty_level.replace(('-'), 0.0,inplace = True)

area_list = percent_over_25_completed_high_school['Geographic Area'].unique()

percent_over_25_completed_high_school.percent_completed_hs = percent_over_25_completed_high_school.percent_completed_hs.astype(float)
percentage_people_below_poverty_level.poverty_rate = percentage_people_below_poverty_level.poverty_rate.astype(float)

high_school_rate = []
poverty_level = []

for i in area_list:
    x = percent_over_25_completed_high_school[percent_over_25_completed_high_school['Geographic Area'] == i]
    y = percentage_people_below_poverty_level[percentage_people_below_poverty_level['Geographic Area'] == i]
    
    high_school_rate.append(sum(x.percent_completed_hs)/len(x))
    poverty_level.append(sum(y.poverty_rate)/len(y))
    
data1 = pd.DataFrame({"area_list": area_list , "area_highschool_ratio": high_school_rate})
new_index = (data1["area_highschool_ratio"].sort_values(ascending=True)).index.values
sorted_data1 = data1.reindex(new_index)

data2 = pd.DataFrame({"area_list": area_list, "poverty_level": poverty_level})
new_index2 = (data2["poverty_level"].sort_values(ascending=True)).index.values
sorted_data2 = data2.reindex(new_index2)

sorted_data1["area_highschool_ratio"] = sorted_data1["area_highschool_ratio"] / max(sorted_data1["area_highschool_ratio"])
sorted_data2["poverty_level"] = sorted_data2["poverty_level"] / max(sorted_data2["poverty_level"])
data = pd.concat([sorted_data1,sorted_data2["poverty_level"]],axis=1)
data.sort_values("area_highschool_ratio", inplace = True)

#visualize

f ,ax1 = plt.subplots(figsize = (20,10))
sns.pointplot(x="area_list",y = "poverty_level", data= data, color= "lime", alpha = 0.8)
sns.pointplot(x="area_list", y = "area_highschool_ratio", data=data, color= "blue", alpha = 0.8)
plt.text(40,0.6,'high school graduate ratio',color='blue',fontsize = 17,style = 'italic')
plt.text(40,0.55,'poverty ratio',color='lime',fontsize = 18,style = 'italic')
plt.xlabel('States',fontsize = 15,color='blue')
plt.ylabel('Values',fontsize = 15,color='blue')
plt.title('High School Graduate  VS  Poverty Rate',fontsize = 20,color='blue')
plt.grid()
    
    
    
    
    

data.head()
sns.jointplot(data.area_highschool_ratio, data.poverty_level, kind = "kde", size = 7)
plt.show()
sns.jointplot("area_highschool_ratio", "poverty_level", data=data, ratio=3, color="r")
plt.show()
f ,ax =plt.subplots(figsize = (5,5))
sns.heatmap(data.corr(), annot = True, linewidths= .5, fmt = ".1f", ax=ax)
plt.show()

kill.head()
sns.boxplot(x="gender", y = "age", hue = "manner_of_death", data=kill, palette="PRGn")
plt.show()

