import pandas as pd

import seaborn as sns

import numpy as np

from matplotlib import pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.


covid_confirmed = pd.read_csv("../input/covid19-confirmed-case-and-death/time_series_covid_19_deaths.csv")

covid_death = pd.read_csv("../input/covid19-confirmed-case-and-death/time_series_covid_19_confirmed.csv")
covid_confirmed.head()
#find total cases for all unique country

unique_country_list = covid_confirmed["Country/Region"].unique()

total_case_list = []

for country in unique_country_list:

    x = covid_confirmed[covid_confirmed["Country/Region"] == country] 

    total = (country, sum(x["4/2/20"]))

    total_case_list.append(total)

co, cs = zip(*total_case_list)

result_data = pd.DataFrame({"country" : co, "total_case" : cs}).sort_values(by="total_case", ascending=False)

most_common_8 = result_data.iloc[:8].reset_index()



#visualization

f, ax = plt.subplots(figsize=(15,10))

sns.barplot(data=most_common_8, x="country", y="total_case", palette="Blues_d")

plt.ylabel("Total Cases", color = "blue", fontsize=18)

plt.xlabel("Countries", color="blue", fontsize=18)



#add number of cases for every columns 

for index, row in most_common_8.iterrows():

    ax.text(row.name, row["total_case"], row["total_case"], ha='center', fontsize=14, alpha=0.7)



plt.title("8 countries with the highest number of confirmed cases as of April 2", fontsize = 20)

plt.show()
#create a data frame that only includes country and time interval columns

cols = covid_confirmed.columns.values.tolist() #get columns

del cols[0], cols[1], cols[1] #remove province, lat and long name from columns list  

data_with_time = pd.DataFrame(columns=cols)



for country in unique_country_list:

    x = list(covid_confirmed[covid_confirmed["Country/Region"] == country].iloc[:, 4:].sum()) #it is pandas.Series so i should convert this to list

    x.insert(0, country)

    s = pd.Series(x, index=data_with_time.columns)

    data_with_time = data_with_time.append(s, ignore_index=True)



data_with_time.sort_values(by="4/2/20", ascending=False, inplace=True) #sort values by last day

melted_data = pd.melt(data_with_time[:8], id_vars=["Country/Region"]) #melt dataframe to use time point



#visualization

f, ax = plt.subplots(figsize=(30,10))

sns.pointplot(data=melted_data, x="variable", y="value", hue="Country/Region")

plt.xticks(rotation=45, fontsize=14)

plt.yticks(fontsize=18)

plt.xlabel("Time interval", fontsize = 17)

plt.ylabel("Confirmed cases", fontsize = 17)

plt.title("Case increase worldwide from 22 January to 2 April in most common 8 countries", fontsize = 25)

plt.grid()

plt.show()


# get countries from sorted data frame

most_common_15 = list(data_with_time["Country/Region"].values[:15])

time_df = data_with_time.iloc[:15, 1:] #get time interval

appearence = [] #to keep appearence date for countries 

for _i in range(len(time_df)): #iterate data frame

    x = time_df.iloc[_i]

    for label, content in x.items(): # iterate inside row

        if content != 0:

            appearence.append(label)

            break



match = zip(most_common_15, appearence) #to see matched values

print(list(match))



result_data_15 = pd.DataFrame({"co" : most_common_15, "appearence" : appearence}).sort_values(by="appearence")



#visualization

f, ax = plt.subplots(figsize=(20,10))

sns.scatterplot(data=result_data_15, x="appearence", y="co", marker="X", s=300, color="red")

plt.xlabel("Appearence", size=18, color="blue")

plt.ylabel("Countries", size=18, color="blue")

plt.title("Virus appearence date in most common 15 countries(sequential)", fontsize = 20)

plt.grid()

plt.show()

covid_death.head()
#new data frame to keep country and sum of death

countries_with_death = pd.DataFrame(columns=["country", "total_death"])



#some countries has more than one province in dataframe so we should group dataframe by countries  

for country in unique_country_list:

    s = covid_death[covid_death["Country/Region"] == country]["4/2/20"].sum()

    pd_s = pd.Series([country, s], index=countries_with_death.columns)

    countries_with_death = countries_with_death.append(pd_s, ignore_index=True)



countries_with_death.sort_values(by="total_death", ascending=False, inplace=True)

final = countries_with_death.reset_index(drop=True)[:15]

final.head(15)



#visualization

f, ax = plt.subplots(figsize=(15,10))

sns.barplot(data=final, x="country", y="total_death", palette="GnBu_d")

plt.xticks(rotation=45) 

plt.xlabel("Countries", size=14)

plt.ylabel("Total Death", size=14)

plt.title("15 countries with the highest covid19 mortality rate", fontsize=20)

plt.show()

#check "Lat" column and if exist drop "Lat" and "Long" 

if "Lat" in covid_death.columns:

    covid_death.drop(["Lat", "Long"], axis=1, inplace=True)



grouped = covid_death.groupby("Country/Region", as_index=False).sum()

grouped.sort_values(by="4/2/20", ascending=False, inplace=True)

allocated = grouped.iloc[:10].reset_index(drop=True)

melted = pd.melt(allocated, id_vars=["Country/Region"]) #to use time point



#visualization

f, ax = plt.subplots(figsize=(25,10))

sns.pointplot(data=melted, x="variable", y="value", hue="Country/Region")

plt.xticks(rotation=45, fontsize=12)

plt.yticks(fontsize=18)

plt.xlabel("Time interval", fontsize=15)

plt.ylabel("Deaths", fontsize=15)

plt.title("Mortality rate in 10 countries with the highest mortality rate from 22 January to 2 April", fontsize=25)

plt.show()