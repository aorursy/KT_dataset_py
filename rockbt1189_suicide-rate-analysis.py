

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# import python libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings="ignore"

import matplotlib.pyplot as plt
df = pd.read_csv("../input/master.csv")

df.head()
#Check the null value(if any)

df.isnull().any()
#HDI foy year is having NULL,lets check the null contribution

df["HDI for year"].isnull().sum()/27820
#Approximate 70% values are null of the column 'HDI for year',So,drop the column from the df.

print(df.shape)

df.drop('HDI for year', axis=1, inplace=True)

print("after drop 'HDI for year' column" ,df.shape)
#Total NO. oF Suicide Case of top 50 Country for the period 1985 to 2016

group = df['suicides_no'].groupby(df['country'])

Total_Suicide_no_by_Country = group.sum().sort_values(ascending=False)

Suicide_no = Total_Suicide_no_by_Country.head(50)

Suicide_no.plot(kind='bar', figsize=(14,4), color='b', alpha=0.5)

plt.title("Total_Suicide_no_by_top 50 Country")
#Total NO. oF Suicide Case by Age Group for the period 1985 to 2016

group = df['suicides_no'].groupby(df['age'])

Total_Suicide_no_by_Age = group.sum().sort_values(ascending=False)

Suicide_no = Total_Suicide_no_by_Age

Suicide_no.plot(kind='bar', figsize=(12, 4), color='r', alpha=0.5)

plt.title("Total_Suicide_no_by_Age")
#Total NO. oF Suicide Case by year for the period 1985 to 2016

group = df['suicides_no'].groupby(df['year'])

Total_Suicide_by_year = group.sum()

Suicide_no = Total_Suicide_by_year

Suicide_no.plot(kind='bar', figsize=(16,4), color='g', alpha=0.5);

plt.title("Total_Suicide_by_year")
#Total NO. oF Suicide Case by year for the period 1985 to 2016

group = df['suicides_no'].groupby(df['generation'])

Total_Suicide_by_generation = group.sum().sort_values(ascending=False)

Suicide_no = Total_Suicide_by_generation

Suicide_no.plot(kind='bar', figsize=(18, 6), color='K', alpha=0.5);

plt.title("Total_Suicide_by_generation")
#Total NO. oF Suicide Case by Gender for the period 1985 to 2016

group = df['suicides_no'].groupby(df['sex'])

Suicide_no = group.sum().sort_values(ascending=True)

Suicide_no.plot(kind='bar', figsize=(16, 4), color='B', alpha=0.5)

plt.title("Total No. oF Suicide by Male and Female")



plt.rcParams["figure.figsize"]=(18,10)



plt.subplot(221)

group = df['suicides_no'].groupby(df['sex'])

Suicide_no = group.sum().sort_values(ascending=True)

#Suicide_no = Suicide_no.head(50)

Suicide_no.plot(kind='bar')



plt.subplot(222)

group = df.groupby(['sex','year']).sum().reset_index()

sns.lineplot(x='year', y='suicides_no', data=group, hue='sex')

plt.title("Year wise Suicides by Gender")



plt.subplot(223)

group = df.groupby(['sex','age']).sum().reset_index()

sns.barplot(x='age', y='suicides_no', data=group, hue='sex')

plt.title("Age group wise Suicides by Gender")





plt.subplot(224)

group = df.groupby(['sex','generation']).sum().reset_index()

sns.barplot(x='generation', y='suicides_no', data=group, hue='sex')

plt.title("generation wise Suicides by Gender")

plt.rcParams["figure.figsize"]=(18,10)



plt.subplot(221)

group = df['suicides/100k pop'].groupby(df['sex'])

Suicide_no = group.sum().sort_values(ascending=True)

#Suicide_no = Suicide_no.head(50)

Suicide_no.plot(kind='bar')



plt.subplot(222)

group = df.groupby(['sex','year']).sum().reset_index()

sns.lineplot(x='year', y='suicides/100k pop', data=group, hue='sex')

plt.title("Year wise Suicides by Gender")



plt.subplot(223)

group = df.groupby(['sex','age']).sum().reset_index()

sns.barplot(x='age', y='suicides/100k pop', data=group, hue='sex')

plt.title("Age group wise Suicides by Gender")





plt.subplot(224)

group = df.groupby(['sex','generation']).sum().reset_index()

sns.barplot(x='generation', y='suicides/100k pop', data=group, hue='sex')

plt.title("generation wise Suicides by Gender")

# Age group wise Suicides over the year

plt.rcParams["figure.figsize"]=(20,4)



plt.subplot(1,2,1)

sns.lineplot(x='year', y='suicides/100k pop', data=df, hue='age')

plt.title("Suicides rate by age group")





plt.subplot(1,2,2)

sns.lineplot(x='year', y='suicides/100k pop', data=df, hue='generation')

plt.title("Suicides rate by generation group")



data = pd.read_csv("https://pkgstore.datahub.io/JohnSnowLabs/country-and-continent-codes-list/country-and-continent-codes-list-csv_csv/data/b7876b7f496677669644f3d1069d3121/country-and-continent-codes-list-csv_csv.csv")

data.head()
data["Country_Name"]= data["Country_Name"].str.split(",", n = 1, expand = True) 

data["Country_Name"]= data["Country_Name"].str.split("(", n = 1, expand = True) 

data.drop(["Continent_Code","Two_Letter_Country_Code","Three_Letter_Country_Code","Country_Number"],axis=1,inplace=True)

data= data.rename({"Continent_Name)": 'Continent',"Country_Name":"country"},axis=1)

data.head()
df2 =pd.merge(df,data, on='country', how='right')

df2.dropna(how='all')

print(df2.shape)

df2.head()
Continent = df2.groupby(["Continent_Name","year"]).sum().reset_index()



Continent["SuicideRatio"] = (Continent["suicides_no"]/Continent["population"])*10000

Continent = Continent[Continent["year"]!=2016]

Continent = Continent[Continent["year"]!=2015]

plt.rcParams["figure.figsize"]=(18,10)



plt.subplot(231)

Africa = Continent[Continent["Continent_Name"]=="Africa"]

sns.lineplot(x="year",y="SuicideRatio",data=Africa)

plt.title("Afica Suicide Rate")



plt.subplot(232)

Asia = Continent[Continent["Continent_Name"]=="Asia"]

sns.lineplot(x="year",y="SuicideRatio",data=Asia)

plt.title("Asia Suicide Rate")



plt.subplot(233)

Europe = Continent[Continent["Continent_Name"]=="Europe"]

sns.lineplot(x="year",y="SuicideRatio",data=Europe)

plt.title("Europ Suicide Rate")



plt.subplot(234)

North_America = Continent[Continent["Continent_Name"]=="North America"]

sns.lineplot(x="year",y="SuicideRatio",data=North_America)

plt.title("North_America Suicide Rate")



plt.subplot(235)

Oceania = Continent[Continent["Continent_Name"]=="Oceania"]

sns.lineplot(x="year",y="SuicideRatio",data=Oceania)

plt.title("Oceania Suicide Rate")





plt.subplot(236)

South_America = Continent[Continent["Continent_Name"]=="South America"]

sns.lineplot(x="year",y="SuicideRatio",data=South_America)

plt.title("South_America Suicide Rate")
#+Gender and Continent wise Suicide trend

Continent = df2.groupby(["Continent_Name","year","sex"]).sum().reset_index()

Continent["SuicideRatio"] = (Continent["suicides_no"]/Continent["population"])*10000

Continent = Continent[Continent["year"]!=2016]

Continent = Continent[Continent["year"]!=2015]

plt.rcParams["figure.figsize"]=(18,10)



plt.subplot(231)

Africa = Continent[Continent["Continent_Name"]=="Africa"]

sns.lineplot(x="year",y="SuicideRatio",data=Africa,hue="sex")

plt.title("Afica Suicide Rate")



plt.subplot(232)

Asia = Continent[Continent["Continent_Name"]=="Asia"]

sns.lineplot(x="year",y="SuicideRatio",data=Asia,hue="sex")

plt.title("Asia Suicide Rate")



plt.subplot(233)

Europe = Continent[Continent["Continent_Name"]=="Europe"]

sns.lineplot(x="year",y="SuicideRatio",data=Europe,hue="sex")

plt.title("Europ Suicide Rate")



plt.subplot(234)

North_America = Continent[Continent["Continent_Name"]=="North America"]

sns.lineplot(x="year",y="SuicideRatio",data=North_America,hue="sex")

plt.title("North_America Suicide Rate")



plt.subplot(235)

Oceania = Continent[Continent["Continent_Name"]=="Oceania"]

sns.lineplot(x="year",y="SuicideRatio",data=Oceania,hue="sex")

plt.title("Oceania Suicide Rate")





plt.subplot(236)

South_America = Continent[Continent["Continent_Name"]=="South America"]

sns.lineplot(x="year",y="SuicideRatio",data=South_America,hue="sex")

plt.title("South_America Suicide Rate")
#Age wise Suicide Rate

Continent = df2.groupby(["Continent_Name","year","age"]).sum().reset_index()



Continent["SuicideRatio"] = (Continent["suicides_no"]/Continent["population"])*10000

Continent = Continent[Continent["year"]!=2016]

Continent = Continent[Continent["year"]!=2015]

plt.rcParams["figure.figsize"]=(18,10)



plt.subplot(231)

Africa = Continent[Continent["Continent_Name"]=="Africa"]

sns.lineplot(x="year",y="SuicideRatio",data=Africa,hue="age")

plt.title("Afica Suicide Rate")



plt.subplot(232)

Asia = Continent[Continent["Continent_Name"]=="Asia"]

sns.lineplot(x="year",y="SuicideRatio",data=Asia,hue="age")

plt.title("Asia Suicide Rate")



plt.subplot(233)

Europe = Continent[Continent["Continent_Name"]=="Europe"]

sns.lineplot(x="year",y="SuicideRatio",data=Europe,hue="age")

plt.title("Europ Suicide Rate")



plt.subplot(234)

North_America = Continent[Continent["Continent_Name"]=="North America"]

sns.lineplot(x="year",y="SuicideRatio",data=North_America,hue="age")

plt.title("North_America Suicide Rate")



plt.subplot(235)

Oceania = Continent[Continent["Continent_Name"]=="Oceania"]

sns.lineplot(x="year",y="SuicideRatio",data=Oceania,hue="age")

plt.title("Oceania Suicide Rate")





plt.subplot(236)

South_America = Continent[Continent["Continent_Name"]=="South America"]

sns.lineplot(x="year",y="SuicideRatio",data=South_America,hue="age")

plt.title("South_America Suicide Rate")