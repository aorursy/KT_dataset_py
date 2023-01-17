# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df=pd.read_csv('/kaggle/input/animal-bites/Health_AnimalBites.csv')

df.head()
df.isnull().sum()
df.describe()
df.columns
df.SpeciesIDDesc.unique()
df.GenderIDDesc.unique()
df.BreedIDDesc.unique()
species = df.SpeciesIDDesc

species = species.dropna() #drop nan values in species feature

speciesOfAnimal = species.unique()

print(speciesOfAnimal)
import seaborn as sns

import matplotlib.pyplot as plt

animal_list = []

for  i in speciesOfAnimal:

    animal_list.append(len(species[species==i]))

ax = sns.barplot(x=speciesOfAnimal, y =animal_list)

plt.title('Number of Species Bite')

plt.xticks(rotation=90)

print(animal_list)
def animal_month(animal,data):

    month_list= ['01','02','03','04','05','06','07','08','09','10','11','12']

    numberOfAnimal = []

    for i in month_list:

        x = df.loc[(df['SpeciesIDDesc']==animal)&(df['bite_date'].str.split('-').str[1]==i)]

        numberOfAnimal.append(len(x))

    ax = sns.barplot(x=month_list,y=numberOfAnimal,palette  = "Blues")

    plt.title(animal + ' bite for 12 month')
animal_month('DOG',df)
animal_month('CAT',df)
animal_month('BAT',df)
count = df.BreedIDDesc.value_counts()

plt.figure(figsize=(15,8))

ax = sns.barplot(x=count[0:10].index,y=count[0:10])

plt.xticks(rotation=20)

plt.ylabel("Number of Bite")

plt.savefig('graph.png')



print(count[0:10].index)
gender = ['MALE','FEMALE']

count_gender = df.GenderIDDesc.value_counts()

plt.figure(figsize= (7,8))

x = sns.barplot(x=gender, y= count_gender[0:2])

plt.ylabel('Number of Bite ')

plt.xticks(rotation = 20)

plt.title('MALE VS FEMALE')

print(count_gender[0:2])
a = df.loc[(df['ResultsIDDesc']=='POSITIVE')]

a = a.loc[:,['bite_date','SpeciesIDDesc','BreedIDDesc','GenderIDDesc','color','ResultsIDDesc']]

print(a)