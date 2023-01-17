# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
percentage_people_below_powerty_level = pd.read_csv("/kaggle/input/fatal-police-shootings-in-the-us/PercentagePeopleBelowPovertyLevel.csv",encoding ="windows-1252")
police_killings_us = pd.read_csv("/kaggle/input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv",encoding ="windows-1252")
share_race_by_city = pd.read_csv("/kaggle/input/fatal-police-shootings-in-the-us/ShareRaceByCity.csv",encoding ="windows-1252")
percent_over_25_completed_high_school = pd.read_csv("/kaggle/input/fatal-police-shootings-in-the-us/PercentOver25CompletedHighSchool.csv",encoding ="windows-1252")
median_household_income = pd.read_csv("/kaggle/input/fatal-police-shootings-in-the-us/MedianHouseholdIncome2015.csv",encoding ="windows-1252")
percentage_people_below_powerty_level.head()
percentage_people_below_powerty_level.info()
percentage_people_below_powerty_level["poverty_rate"].value_counts()
percentage_people_below_powerty_level["poverty_rate"].replace("-",0.0,inplace=True)
percentage_people_below_powerty_level["poverty_rate"] = percentage_people_below_powerty_level["poverty_rate"].astype(float)
area_list = percentage_people_below_powerty_level["Geographic Area"].unique()
poverty_rate_list = []
for i in area_list:
    x = percentage_people_below_powerty_level[percentage_people_below_powerty_level["Geographic Area"] == i]
    poverty_rate = x.poverty_rate.sum()/len(x)
    poverty_rate_list.append(poverty_rate)
new_data = pd.DataFrame({"area_list":area_list,"poverty_rate":poverty_rate_list})
new_index = new_data["poverty_rate"].sort_values(ascending=False).index.values
sorted_data = new_data.reindex(new_index)
#Visulisation 
plt.figure(figsize=(15,10))
sns.barplot(x=sorted_data["area_list"],y=sorted_data["poverty_rate"])
plt.xlabel("Eyaletler")
plt.ylabel("Povery Rate")
plt.title("Eyaletler Fakirlik Oranı")
plt.show()
police_killings_us.head()
police_killings_us.info()
police_killings_us["name"].value_counts()
seperate = police_killings_us["name"][police_killings_us["name"] != "TK TK"].str.split()
a = list(seperate)
k = []
for i in range(len(a)):
    k.append(a[i][0])
name_data = pd.DataFrame({"name":k})
count = name_data["name"].value_counts()
count.sort_values(ascending=False,inplace=True)
last_name = count.head(15)
#Visualize

plt.figure(figsize=(15,10))
sns.barplot(x=last_name.index,y=last_name.values,palette="rocket")
plt.xlabel("Names")
plt.ylabel("Frequency")
plt.title("Frequency of Names")
plt.show()
percent_over_25_completed_high_school.head()
percent_over_25_completed_high_school.info()
percent_over_25_completed_high_school["percent_completed_hs"].value_counts()
percent_over_25_completed_high_school["percent_completed_hs"].replace("-","NaN",inplace=True)
school_data = percent_over_25_completed_high_school.dropna()
school_data["percent_completed_hs"]=school_data["percent_completed_hs"].astype(float)
state_list = list(school_data["Geographic Area"].unique())
state_list
graduate = []
for each in state_list:
    a = school_data[school_data["Geographic Area"] == each]
    rate = a["percent_completed_hs"].sum()/len(a)
    graduate.append(rate)
last_data =pd.DataFrame({"State":state_list,"Graduate":graduate})
last_data.sort_values("Graduate",ascending=True,inplace=True)
last_data
#Visualization

plt.figure(figsize=(15,10))
sns.barplot(x=last_data["State"],y=last_data["Graduate"])
plt.xlabel("States")
plt.ylabel("Graduate")
plt.title("Graduate Rates in States")
plt.savefig("Graphic.png")
plt.show()
share_race_by_city.head()
share_race_by_city.info()
share_race_by_city.replace("(X)",0.0,inplace=True)
for each in a:
    if each == "Geographic area":
        pass
    elif each == "City":
        pass
    else:
        share_race_by_city[each] = share_race_by_city[each].astype(float)

share_race_by_city.head()
area_list = list(share_race_by_city["Geographic area"].unique())
white_rate = []
black_rate = []
native_american_rate = []
asian_rate = []
hispanic_rate = []
for each in area_list:
    each_list = share_race_by_city[share_race_by_city["Geographic area"] == each]
    w_rate = each_list["share_white"].sum()/len(each_list)
    b_rate = each_list["share_black"].sum()/len(each_list)
    na_rate = each_list["share_native_american"].sum()/len(each_list)
    a_rate = each_list["share_asian"].sum()/len(each_list)
    h_rate = each_list["share_hispanic"].sum()/len(each_list)
    white_rate.append(w_rate)
    black_rate.append(b_rate)
    native_american_rate.append(na_rate)
    asian_rate.append(a_rate)
    hispanic_rate.append(h_rate)

white_rate

demographic_data = pd.DataFrame({"Geographic area":area_list,"share_white":white_rate,"share_black":black_rate,"share_native_american":native_american_rate,"share_asian":asian_rate,"share_hispanic":hispanic_rate})
demographic_data.head()
#Visualization

plt.figure(figsize=(15,10))
sns.barplot(x=demographic_data["share_white"],y=demographic_data["Geographic area"],color="red",alpha=0.5)
sns.barplot(x=demographic_data["share_black"],y=demographic_data["Geographic area"],color="green",alpha=0.7)
sns.barplot(x=demographic_data["share_native_american"],y=demographic_data["Geographic area"],color="blue",alpha=0.8)
sns.barplot(x=demographic_data["share_asian"],y=demographic_data["Geographic area"],color="yellow",alpha=0.9)
sns.barplot(x=demographic_data["share_hispanic"],y=demographic_data["Geographic area"],color="purple",alpha=0.5)
plt.xlabel("Share Rates")
plt.ylabel("States")
plt.title("Demographic Rate in States")
plt.show()
education = last_data.sort_values("State")
poverty = new_data.sort_values("area_list")
education["Graduate"] = education["Graduate"]/max(education["Graduate"])
poverty["poverty_rate"] = poverty["poverty_rate"]/max(poverty["poverty_rate"])
compare_data = pd.concat([education,poverty["poverty_rate"]],axis=1)
compare_data.sort_values("Graduate",ascending=True,inplace=True)
compare_data
#Visualization
plt.figure(figsize=(15,10))
sns.pointplot(x="State",y="poverty_rate",data=compare_data,color="red",label="Poverty")
sns.pointplot(x="State",y="Graduate",data=compare_data,color="blue",label="Graduate")
plt.text(3,0.3,"Graduate",color="blue",fontsize=14,style="oblique")
plt.text(3,0.35,"Poverty Rate",color="red",fontsize=14,style="oblique")
plt.xlabel("States")
plt.ylabel("Rate")
plt.show()