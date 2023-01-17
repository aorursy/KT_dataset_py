#setting up all the essentials



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from datetime import datetime #date and time related manipulation



#data visualizations

import matplotlib.pyplot as plt

import seaborn as sns



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#import Seattle Pet Liecense and get an idea of how the data looks

pets = pd.read_csv("../input/seattle-pet-licenses/Seattle_Pet_Licenses.csv")

pets.head()
#Rename columns so that they are more easily accesible to reference in future code

pets.columns = ["issue_date","id_number", "name","species","breed","second_breed","zip_code"]

pets.columns
#Check to make sure each column is the correct data type

pets.dtypes
#Change the Issue date into a datetype

count = 0;

year = []

for date in pets['issue_date']:

    date = datetime.strptime(date, "%B %d %Y")

    pets['issue_date'][count] = date

    year.append(date.year)

    count = count + 1



#create a new column for year only to easily match with Seattle dataset

pets.insert(1, "year", year, True)



pets
#import file and view data 

seattle = pd.read_csv("../input/seattle-annual-stats/City_Annual_Stats.csv")



#Keep only relevant columns with new names  

seattle = seattle[["Year_", "Households", "Total_Population"]]

seattle.columns = ["year", "households","population"]



#drop last 2 rows that do not provide any information

seattle = seattle.drop(seattle.index[[20, 21]])



#drop the first row since the year is earlier than the rest of the sequence and does not match the pets dataset

seattle = seattle.drop(seattle.index[0])



#Change all columns to integers

seattle = seattle.astype('int32')



#check data types    

seattle.dtypes



#get a look at cleaned data

seattle
# visualize distribution of pets by species

plt.figure(figsize=(8, 4))

sns.countplot(pets['species'], palette='RdBu')





# count each species

dogs, cats, goats, pigs = pets['species'].value_counts()

print('Number of Dogs: ', dogs)

print('Number of Cats : ', cats)

print('Number of Goats: ', goats)

print('Number of Pigs : ', pigs)

print('')

print('% of Dogs', round(dogs / len(pets) * 100, 2), '%')

print('% of Cats', round(cats / len(pets) * 100, 2), '%')

print('% of Goats', round(goats / len(pets) * 100, 2), '%')

print('% of Pigs', round(pigs / len(pets) * 100, 2), '%')
#Group data by species to be analysed be species type

species = pets.groupby('species')

for specie, group in species:

    print(specie)

    print(group)
#visualize the amount of pets registered across the years

plt.figure(figsize=(8, 4))

sns.countplot(pets['year'], palette='RdBu')
#Group by year to be analyze data through time

by_year = pets.groupby('year')

for year, group in by_year:

    print(year)

    print(group)
#Store the sum of each pet for each year in a new data frame to be graphed in a line graph

count = list(by_year.count()['issue_date'])

year= list(by_year.groups.keys())

data = [year,count]



count_by_year = pd.DataFrame(data).T

count_by_year.columns = ["year", "count"]

count_by_year = count_by_year[:-2]

count_by_year
#plot the total population, household and pets all in a single graph 

plt.figure(figsize=(10, 8))

plt.plot('year', 'population', data=seattle, alpha = 0.9, label = "Total Population")

plt.plot('year', 'households', data=seattle, alpha = 0.9, label = "Total Households")

plt.plot('year', 'count', data=count_by_year, label = "Registered Pets")



plt.legend(loc=2, ncol=10)

plt.title('Population and Households in Seattle across the years')





plt.show()
#plot the number of pets against the total population

plt.figure(figsize=(15, 3))

x = seattle['population']

y = count_by_year['count']

plt.scatter(x, y)

plt.xlabel("Total Population")

plt.ylabel("Number of Pets Registered")

plt.show()

plt.figure(figsize=(15, 3))



#plot the number of pets against the number of households

x = seattle['households']

y = count_by_year['count']

plt.xlabel("Number of Households")

plt.ylabel("Number of Pets Registered")

plt.scatter(x, y)

plt.show()