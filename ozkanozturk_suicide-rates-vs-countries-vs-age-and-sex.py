# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Read data from csv file



data = pd.read_csv("/kaggle/input/who-suicide-statistics/who_suicide_statistics.csv")



# Show columns of data



data.columns
# Show first 5 rows



data.head()
# Show last 5 rows



data.tail()
# Rows and columns count

data.shape
# Show some info on data



data.info()
print(data.describe())
data.head()
data.population.dropna(inplace=True)

data.suicides_no.dropna(inplace=True)
# Suicides vs Ages

age_list = data.age.value_counts().index.values # creating a list as to incldue age counts

total_suicide=[] # creating list for each age counts

# for loop to calculate suicide qty for each age

for each in age_list:

  age_filter = data["age"]==each

  suicide_sum_each = data[age_filter].suicides_no.sum()

  total_suicide.append(suicide_sum_each)

total_suicide

# creating new DataFrame from "age_list" and "total_suicide"

new_data = pd.DataFrame({

  "age_list":age_list,

  "total_suicide":total_suicide})

new_data
# adding new column to new_data

suicide_rate=[each*100/(new_data.total_suicide.sum()) for each in new_data.total_suicide]

new_data["suicide_rate"] = suicide_rate

new_data
# sorting and reindexing new_data

# sorting data

new_index = new_data.total_suicide.sort_values(ascending=False).index.values

# reindexing data

sorted_new_data = new_data.reindex(new_index)

sorted_new_data
# VISUALIZATION



# barplot

plt.figure(figsize=(15,10))

sns.barplot(x=sorted_new_data.age_list, y=sorted_new_data.suicide_rate )

plt.xticks(rotation= 45)

plt.xlabel("People age intervals", fontsize=15, color="blue")

plt.ylabel("Suicide rates = suicide / total suicide x 100", fontsize=15, color = "blue")

plt.title("Suicide rates of people in different ages in glob", fontsize = 15, color="b")

plt.grid()
# VISUALIZATION



# pieplot

sizes = sorted_new_data.total_suicide

labels = sorted_new_data.age_list.values

colors = ['grey','blue','red','yellow','green','brown'] 

explode = [0.1,0.1,0.1,0.1,0.1,0.1]



plt.figure(figsize = (10,10))

plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%')

plt.title('Suicided People According to Ages',color = 'blue',fontsize = 15)  

#   SUICIDE RATES ACCORDING TO COUNTRIES 



countries = data.country     # getting countries from data

suicides = data.suicides_no   # getting suicides from data

# creating new dataframe from countries and suicides

new_data1 = pd.DataFrame({"countries":countries, "suicides":suicides})

new_data1.suicides.dropna(inplace = True)   # dropping NaN values in suicides column

new_data1.head()
# creating list of countries

country_list = new_data1.countries.value_counts().index

country_list
# creating an empty list before for each loop

country_suicide = []



# Calculating sum of suicides for each countries in data and adding them into list

for each in country_list:

  filter1 = new_data1["countries"]== each

  x = new_data1[filter1].suicides.sum()

  country_suicide.append(x)



# creating new data frame

filtered_data1 = pd.DataFrame({

  "country_list":country_list,

  "country_suicide":country_suicide})



filtered_data1.head()
# sorting data and creating new index

new_index1 = filtered_data1.country_suicide.sort_values(ascending=False).index

# creating new data frame by re-indexing the old data

sorted_data1 = filtered_data1.reindex(new_index1)

sorted_data1.head()
# calculating suicide rate for countries

x = sorted_data1.country_suicide.sum()

country_suicide_rate = sorted_data1.country_suicide/x*1000

new_column = pd.DataFrame({"country_suicide_rate":country_suicide_rate})



# adding the new_column data to data frame

sorted_data1 = pd.concat([sorted_data1, new_column], axis=1)

sorted_data1.head()
# filtering the data as to have suicide rates > 1

sorted_data2 = sorted_data1[sorted_data1["country_suicide_rate"]>1]

sorted_data2.info()
# filtering the data as to have suicide rates < 1

sorted_data3 = sorted_data1[sorted_data1["country_suicide_rate"]<1]

sorted_data3.info()
#visualization for suicide rates > 1

plt.figure(figsize = (50,20))

sns.barplot(x=sorted_data2.country_list, y=sorted_data2.country_suicide_rate)

plt.xticks(rotation= 45, fontsize = 20)

plt.yticks(fontsize = 20)

plt.xlabel("Countries", fontsize = 30, color = "b")

plt.ylabel("Suicide rate = Country suicides/Global Suicides*1000", fontsize = 30, color = "b")

plt.title("COUNTRY SUICIDE RATES,  RATES GREATER THAN 1 ",fontsize = 30, color = "b")
#visualization for suicide rates < 1

plt.figure(figsize = (50,20))

sns.barplot(x=sorted_data3.country_list, y=sorted_data3.country_suicide_rate)

plt.xticks(rotation= 45, fontsize = 20)

plt.yticks(fontsize = 20)

plt.xlabel("Countries", fontsize = 30, color = "b")

plt.ylabel("Suicide rate = Country suicides/Global Suicides*1000", fontsize = 30, color = "b")

plt.title("COUNTRY SUICIDE RATES,  RATES LESS THAN 1",fontsize = 30, color = "b")
#  TURKEY'S DATA

filter_turkey = data["country"]=="Turkey"

data_turkey = data[filter_turkey]



# boxplot

plt.figure(figsize=(10,10))

sns.boxplot(x="age", y="suicides_no" , hue="sex", data = data_turkey)

plt.grid()

plt.xlabel("Age interval", fontsize=15, color = "b")

plt.ylabel("Suicide numbers", fontsize=15, color = "b")

plt.title("Suicides vs age and sex in Turkey", color ="b",fontsize = 15)
# swarm plot

plt.figure(figsize=(7,5))

sns.swarmplot(x="age", y="suicides_no" , hue="sex", data = data_turkey)

plt.grid()

plt.xlabel("Age interval", fontsize=15, color = "b")

plt.ylabel("Suicide numbers", fontsize=15, color = "b")

plt.title("Suicides vs age and sex in Turkey", color ="b",fontsize = 15)
# pair plot

plt.figure(figsize=(15,15))

sns.pairplot(data_turkey)

plt.grid()

plt.show()
# swarm plot USA

plt.figure(figsize=(7,5))

sns.swarmplot(x="age", y="suicides_no" , hue="sex", data = data[data["country"]=='United States of America'])

plt.grid()

plt.xlabel("Age interval", fontsize=15, color = "b")

plt.ylabel("Suicide numbers", fontsize=15, color = "b")

plt.title("Suicides vs age and sex in USA", color ="b",fontsize = 15)
# swarm plot UK

plt.figure(figsize=(7,5))

sns.swarmplot(x="age", y="suicides_no" , hue="sex", data = data[data["country"]=='United Kingdom'])

plt.grid()

plt.xlabel("Age interval", fontsize=15, color = "b")

plt.ylabel("Suicide numbers", fontsize=15, color = "b")

plt.title("Suicides vs age and sex in UK", color ="b",fontsize = 15)
# swarm plot Russian Federation

plt.figure(figsize=(7,5))

sns.swarmplot(x="age", y="suicides_no" , hue="sex", data = data[data["country"]=='Russian Federation'])

plt.grid()

plt.xlabel("Age interval", fontsize=15, color = "b")

plt.ylabel("Suicide numbers", fontsize=15, color = "b")

plt.title("Suicides vs age and sex in Russian Federation", color ="b",fontsize = 15)
# swarm plot Japan

plt.figure(figsize=(7,5))

sns.swarmplot(x="age", y="suicides_no" , hue="sex", data = data[data["country"]=='Japan'])

plt.grid()

plt.xlabel("Age interval", fontsize=15, color = "b")

plt.ylabel("Suicide numbers", fontsize=15, color = "b")

plt.title("Suicides vs age and sex in Japan", color ="b",fontsize = 15)
# swarm plot Sweden

plt.figure(figsize=(7,5))

sns.swarmplot(x="age", y="suicides_no" , hue="sex", data = data[data["country"]=='Sweden'])

plt.grid()

plt.xlabel("Age interval", fontsize=15, color = "b")

plt.ylabel("Suicide numbers", fontsize=15, color = "b")

plt.title("Suicides vs age and sex in Sweden", color ="b",fontsize = 15)
# swarm plot Germany

plt.figure(figsize=(7,5))

sns.swarmplot(x="age", y="suicides_no" , hue="sex", data = data[data["country"]=='Germany'])

plt.grid()

plt.xlabel("Age interval", fontsize=15, color = "b")

plt.ylabel("Suicide numbers", fontsize=15, color = "b")

plt.title("Suicides vs age and sex in Germany", color ="b",fontsize = 15)