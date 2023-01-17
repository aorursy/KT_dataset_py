import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

import seaborn as sns
india=pd.read_csv("../input/global-hospital-beds-capacity-for-covid19/hospital_beds_per_india_v1.csv")

Usa=pd.read_csv("../input/global-hospital-beds-capacity-for-covid19/hospital_beds_USA_v1.csv")
india.head()
Usa.head()
india.describe()
Usa.describe()
india.info()
Usa.info()
india.drop(india.columns[india.isnull().all()],inplace=True, axis=1)

Usa.drop(Usa.columns[Usa.isnull().all()],inplace=True, axis=1)

india.beds.iloc[30:31]=0.971725
#To get the unique values from column

print(india.type.unique())

print(india.measure.unique())

print(india.year.unique())
#To get the unique values from column

print(Usa.type.unique())

print(Usa.measure.unique())

print(Usa.year.unique())
#Dropping Useless features

india=india.drop(['country','measure','source','source_url','lat','lng'],axis=1)
india.info()
Usa=Usa.drop(['country','measure','source','source_url','lat','lng'],axis=1)
Usa.info()
#Checking is their any null values or not

india.isnull().sum()
Usa.isnull().sum()
india["Total_beds"]=(india["population"]/1000)*india["beds"] #Calculating total beds as we know beds are per 1000

Usa["Total_beds"]=(Usa["population"]/1000)*Usa["beds"]

india["beds_percent"]=(india.Total_beds/india.population)*100 #Calculating bed's Percentage

Usa["beds_percent"]=(Usa.Total_beds/Usa.population)*100
india.info()
Usa.info()
#Plotting the year and total number of beds in India

india[["Total_beds","year"]].groupby('year').sum().Total_beds.plot(kind="bar",color="red")

plt.title("India")

plt.ylabel("Total Number of bed")

plt.xlabel("Year")
#Plotting Year and its count of beds in USA

Usa[["Total_beds","year"]].groupby('year').sum().Total_beds.plot(kind="bar",color="blue")

plt.ylabel("Total Number of bed")

plt.xlabel("Year")

plt.title("USA")
india[["beds_percent","year"]].groupby('year').sum().beds_percent.plot(kind="line",color="red")

plt.ylabel("Bed Percent")

plt.xlabel("Year")

plt.title("India")
Usa[["beds_percent","year"]].groupby('year').sum().beds_percent.plot(kind="line",color="navy")

plt.ylabel("Bed Percent")

plt.xlabel("Year")

plt.title("USA")
f, axes = plt.subplots(1,1,figsize=(18,5))

sns.lineplot(y="population", x= 'state', data=india )

plt.title('Population vs State Plot for India')

plt.show()
f, axes = plt.subplots(1,1,figsize=(18,5))

sns.lineplot(y="population", x= 'state', data=Usa )

plt.title('Population vs State Plot for Usa')

plt.show()
f, axes = plt.subplots(1,1,figsize=(18,5))

sns.lineplot(y="population", x= 'Total_beds', data=india )

plt.title('Population vs Total beds Plot for Usa')

plt.show()
f, axes = plt.subplots(1,1,figsize=(18,5))

sns.lineplot(y="population", x= 'Total_beds', data=Usa )

plt.title('Population vs Total beds Plot for Usa')

plt.show()
#Type vs Total beds

Usa[["Total_beds","type"]].groupby('type').sum().Total_beds.plot(kind="bar",color="navy")

plt.title("USA")

plt.xlabel("Type")

plt.ylabel("Beds")
#Pie char for total beds available in india

temp=sum(india.Total_beds)

(india[["Total_beds","year"]].groupby('year').sum().Total_beds/temp).plot(kind="pie",shadow=True,autopct='%1.1f%%',radius=1,explode=[0,0.1,0.1,0.1,0])

plt.title("Total Bed Avaliable India",loc="center")
#Pie char for total beds available in USA

Usa[["Total_beds","type"]].groupby('type').sum().Total_beds.plot(kind="pie",shadow=True,autopct='%0.1f%%',radius=1,explode=[0,0.1,0.1,0.1],startangle=90)

plt.title("Types of Bed Avaliable USA")
#Box plot for type and year

Value=Usa.groupby(['type','year'])['beds'].mean()

output=Value.reset_index()

plt.figure(figsize = (6, 6))

ax = sns.boxplot(x='type', y='beds', data=output)

ax.set_title('Bed capacities according to types in USA')

plt.xticks()
plt.title("India")

plt.xlabel("Year")

plt.ylabel("Population(Millions)")

india[["population","year"]].groupby('year').sum().population.plot(kind="bar",color="blue")
plt.title("Usa")

plt.xlabel("Year")

plt.ylabel("Population(Millions)")

Usa[["population","year"]].groupby('year').sum().population.plot(kind="bar",color="navy")
#India vs USA Population

plt.plot(sorted(Usa.year.unique()),Usa[["year","population"]].groupby('year').sum().population,'-o',label="USA Population")

plt.plot(sorted(india.year.unique()),india[["year","population"]].groupby('year').sum().population,'g-D',label="India Population")

plt.ylabel("Population")

plt.xlabel("Year")

plt.title("India vs USA Population")

plt.legend()
#India vs USA TOTAL BEDS

plt.plot(sorted(Usa.year.unique()),Usa[["year","Total_beds"]].groupby('year').sum().Total_beds,'-o',label="USA Total bed")

plt.plot(sorted(india.year.unique()),india[["year","Total_beds"]].groupby('year').sum().Total_beds,'-o',label="India Total bed")

plt.ylabel("Total bed")

plt.xlabel("Year")

plt.title("India vs USA Total bed")

plt.legend()