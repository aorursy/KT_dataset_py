import numpy as np 

import pandas as pd 



import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as ex



%matplotlib inline

sns.set_style("darkgrid")
df = pd.read_csv("/kaggle/input/daily-temperature-of-major-cities/city_temperature.csv")
df.head()
df.dtypes
df.isna().sum()
df.describe()
print (f"Unique Years : {df.Year.unique()}")
mask = df.Year == 200

df.loc[mask,"Year"] = 2000



mask = df.Year == 201

df.loc[mask,"Year"] = 2010
df.Day.unique()
mask = df.Day == 0

df.loc[mask,"Day"] = 1
print (f"Total Regions : {df.Region.nunique()}")

print (f"Total Countries : {df.Country.nunique()}")

print (f"Total Cities : {df.City.nunique()}")



print (f"\nWe have data of total {df.Year.nunique()} years starting from {df.Year.min()} to {df.Year.max()}.")

print (f"The temperature ranges from {df.AvgTemperature.min()} ᵒC to {df.AvgTemperature.max()} ᵒC")
regions = df.Region.value_counts()



plt.figure(figsize=(12,4))

sns.barplot(regions.values,regions.index,color="#3498db")

plt.title("Data Amount From Various Regions");
countries = df.Country.value_counts()



plt.figure(figsize=(12,36))

sns.barplot(countries.values,countries.index,color="#3498db")

plt.title("Data Amount From Various Countries");
cities = df.City.value_counts().sort_values(ascending=False)

fig,axes = plt.subplots(2,1,figsize=(14,18))



ax = sns.barplot(cities.head(25).index,cities.head(25).values,color="#3498db",ax=axes[0])

ax.set_xticklabels(ax.get_xticklabels(),rotation=60)

ax = sns.barplot(cities.tail(25).index,cities.tail(25).values,color="#3498db",ax=axes[1])

ax.set_xticklabels(ax.get_xticklabels(),rotation=60);
years = df['Year'].value_counts()



plt.figure(figsize=(14,6))

sns.barplot(years.index,years.values,color="#3498db")

plt.title("Number Of Records For Every Year")

plt.xticks(rotation=45);
months = df.Month.value_counts()



plt.figure(figsize=(14,6))

sns.barplot(months.index,months.values,color="#3498db")

plt.title("Number Of Records For Every Month");
day = df.Day.value_counts()



plt.figure(figsize=(14,6))

sns.barplot(day.index,day.values,color="#3498db")

plt.title("Number Of Records For Every Day");
plt.figure(figsize=(14,6))

sns.distplot(df.AvgTemperature)

plt.title("Temperature Distribution");
temp = df[['Year','AvgTemperature']]

group = temp.groupby("Year")
mean_temp = group.mean()

plt.figure(figsize=(14,5))

sns.lineplot(mean_temp.index,mean_temp.AvgTemperature,color="#2ecc71")

plt.xticks(mean_temp.index,rotation=90)

plt.title("Average Temperature For Every Year");



max_temp = group.max()

plt.figure(figsize=(14,5))

sns.lineplot(max_temp.index,max_temp.AvgTemperature,color="#2ecc71")

plt.xticks(max_temp.index,rotation=90)

plt.title("Maximum Temperature For Every Year");



min_temp = group.min()

plt.figure(figsize=(14,5))

sns.lineplot(min_temp.index,min_temp.AvgTemperature,color="#2ecc71")

plt.xticks(min_temp.index,rotation=90)

plt.title("Minimun Temperature For Every Year");
temp = df[['Month','AvgTemperature']]

group = temp.groupby("Month")
mean_temp = group.mean()

plt.figure(figsize=(14,5))

sns.lineplot(mean_temp.index,mean_temp.AvgTemperature,color="#2ecc71")

plt.xticks(mean_temp.index,rotation=90)

plt.title("Average Temperature For Every Month");



max_temp = group.max()

plt.figure(figsize=(14,5))

sns.lineplot(max_temp.index,max_temp.AvgTemperature,color="#2ecc71")

plt.xticks(max_temp.index,rotation=90)

plt.title("Maximum Temperature For Every Month");



min_temp = group.min()

plt.figure(figsize=(14,5))

sns.lineplot(min_temp.index,min_temp.AvgTemperature,color="#2ecc71")

plt.xticks(min_temp.index,rotation=90)

plt.title("Minimun Temperature For Every Month");
temp = df[['Region','Country','City','AvgTemperature']]

group = temp.groupby(['Region'])
min_temp = group.mean()

plt.figure(figsize=(14,5))

sns.barplot(min_temp.index,min_temp.AvgTemperature,color="#2ecc71")

plt.xticks(rotation=60)

plt.title("Average Temperature For Every Region");



rows = []

for region in group.groups.keys():

    g = group.get_group(region)

    rows.append(g[g.AvgTemperature.max() == g.AvgTemperature].values[0])

    

t = pd.DataFrame(rows,columns=['Region','Country','City','Temp'])



plt.figure(figsize=(14,5))

p = sns.barplot(t.Region,t.Temp,color="#2ecc71")

plt.xticks(rotation=60)

plt.title("Maximum Temperature For Every Region");



for index, row in t.iterrows():

    p.text(index,35, f"{row.City}, {row.Country}", color='#333', ha="center",rotation=90)

    

    



rows = []

for region in group.groups.keys():

    g = group.get_group(region)

    rows.append(g[g.AvgTemperature.min() == g.AvgTemperature].values[0])

    

t = pd.DataFrame(rows,columns=['Region','Country','City','Temp'])



plt.figure(figsize=(14,5))

p = sns.barplot(t.Region,t.Temp,color="#2ecc71")

plt.xticks(rotation=60)

plt.title("Minimum Temperature For Every Region");



for index, row in t.iterrows():

    p.text(index,-65, f"{row.City}, {row.Country}", color='#333', ha="center",rotation=90)

    