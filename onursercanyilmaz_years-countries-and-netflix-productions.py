#Import libraries



import numpy as np 

import pandas as pd 



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



import matplotlib.pyplot as plt

import seaborn as sns

from subprocess import check_output
df = pd.read_csv('/kaggle/input/netflix-shows/netflix_titles.csv')

df

                 


df.info()
df.columns
print(df[['type', 'listed_in']])
f, ax = plt.subplots(figsize=(18,18))

sns.heatmap(df.corr(), annot=True, linewidths = 5, fmt='.1f', ax=ax)

df.corr()
#Unique types of productions

df['type'].unique()
#Set Movie types in "Movie"

Movie = df[df.type == 'Movie']

Movie.count()
#Set TV Show types in "Series"

Series = df[df.type == 'TV Show']

Series.count()

Series.listed_in.unique()
year = pd.DataFrame(df["release_year"])

year
year.plot.kde(bw_method=0.04)

plt.grid(True,color="#c3c0b2")

plt.title("Year by year released TV Show + Movie Density", color="#0b77a2")

plt.xlabel("Years", color="#0b77a2")

plt.ylabel("DENSITY",color="#0b77a2")

plt.show()
#Year by year released TV Show + Movie which Netflix has

df.release_year.plot(kind="hist", bins=200, figsize=(10,10), color="#c52c2c")

plt.grid(True,color="#c3c0b2")

plt.title("Year by year released TV Show + Movie which Netflix has", color="#0b77a2")

plt.xlabel("Years", color="#0b77a2")

plt.ylabel("Number of Released Movie + TV Shows",color="#0b77a2")

plt.show()
plt.figure(figsize=(8,8))

labels = df["release_year"].unique()

values = df["release_year"].value_counts()



plt.pie(values, labels=labels,autopct="%.1f%%")



plt.show()
df.info()
plt.figure(figsize=(10,10))

plt.scatter(df.release_year,df.show_id,  color='red', alpha=0.5)

plt.xlabel('Release Year')

plt.ylabel('Show ID')

plt.title('Correlation between Show ID & Release Years of Productions', color="green")

plt.grid(True)

plt.show()
#df.country.unique() --too long
#How many production released in countries

df.country.value_counts()
#Determine frequencies of productions country by country

US = df[df.country == 'United States']

us = US.show_id.count()

UK = df[df.country == 'United Kingdom']

uk = UK.show_id.count()

CAN = df[df.country == 'Canada']

can = CAN.show_id.count()

SPA = df[df.country == 'Spain']

spa = SPA.show_id.count()

GER = df[df.country == 'Germany']

ger = GER.show_id.count()

IND = df[df.country == 'India']

ind = IND.show_id.count()

CHI = df[df.country == 'China']

chi = CHI.show_id.count()



#Create dictionary

#dict = {'United States':us,'United Kingdom':uk,'India':ind,'Canada':can,'Spain':spa,'Germany':ger,'China':chi  }

dict = {"country":['United States','United Kingdom','India','Canada','Spain','Germany','China'], "frq":[us,uk,ind,can,spa,ger,chi]}
#Create dataframe from dictionary

FRQ = pd.DataFrame({"country":['United States','United Kingdom','India','Canada','Spain','Germany','China'], "frq":[us,uk,ind,can,spa,ger,chi]})

FRQ
#Visualize the frequencies with histogram



FRQ.plot.bar(x="country", y="frq", rot=45,color="#a1332b",figsize=(15,5))

plt.xlabel("Country")

plt.ylabel("Number of Production")

plt.title("Number of Productions Country by Country")

plt.grid(True, color="#787777")

plt.show()
#Visualize the frequencies with pie

plt.figure(figsize=(8,8))



labels = FRQ.country

values = FRQ.frq



plt.pie(values, labels=labels,autopct="%.1f%%")



plt.show()