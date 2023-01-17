# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#reading in the dataframe
spl = pd.read_csv("../input/seattle_pet_licenses.csv")
#getting the idea of structure of dataframe
print(spl.info(),"\n\n")
print("The dimensions of the dataset are:",spl.shape)
#taking a peek 
spl.sample(5)
print("The count of null values for columns in dataset is:\n")
print(spl.isnull().sum())
#From this is evident that columns animals_s_name, license_number, secndary_breed and zip_codes have some null values
column_name = 'secondary_breed'
if column_name in list(spl.columns):
    spl.drop(columns = ['secondary_breed'], inplace = True)    
#There are 1357 null values in column 'animal_s_name' dropping those columns, and 158 null values in column 'zip_code' and therefore removing those values 
spl = spl.dropna(axis =0, how = 'any', subset = ['animal_s_name','zip_code'])
print("The count of null values for columns in dataset is:\n")
print("Column\t\t\t Count")
print(spl.isnull().sum(), "\n")
print("The dimensions of the dataset now are:", spl.shape)
spl = spl.rename(columns = {'species':'Species'})
f,ax = plt.subplots(figsize = (10,8))
sns.countplot(x = 'Species', data = spl,hue = 'Species', ax = ax)
plt.ylabel("Count")
plt.xlabel('')
ax.set_xticks([])
sns.despine(left = True)
#let's start by creating a dataframe having names and their frequency
array = spl.animal_s_name.value_counts().reset_index().values
common_pet_names = pd.DataFrame(data = array, columns = ['Names','Frequency'])
common_pet_names_t20 = common_pet_names.head(20)
fig, ax = plt.subplots(figsize=(11.7,8.7))
plt.xticks(rotation = 45)
sns.set_context("notebook")
sns.barplot(x = 'Names', y = 'Frequency',data = common_pet_names_t20, ax= ax)
plt.xlabel('')
sns.despine(bottom = True)

grouped_by_species = spl.groupby(['Species','primary_breed'])
grouped_by_species_count = grouped_by_species.count()
grouped_by_species_count.drop(columns = ['license_issue_date','license_number','zip_code'], inplace = True)
grouped_by_species_count = grouped_by_species_count.reset_index()
grouped_by_species_count = grouped_by_species_count.rename(columns = {'Species':'Species', 'primary_breed':'Primary_Breed', 'animal_s_name':'Count'})
grouped_by_species_cats = grouped_by_species_count[grouped_by_species_count.Species == 'Cat']
grouped_by_species_dogs = grouped_by_species_count[grouped_by_species_count.Species == 'Dog']
print(grouped_by_species_cats.sort_values('Count',ascending = False).head(10),"\n\n")
print(grouped_by_species_dogs.sort_values('Count',ascending = False).head(10))
#Plotting most popular breeds of Cats
fig, ax = plt.subplots(figsize = (10,8))
plot =sns.barplot(x ='Count' , y ='Primary_Breed' , data = grouped_by_species_cats.sort_values('Count',ascending = False).head(10))
plt.ylabel('')
plt.xlabel('')
ax.set_xticks([])
sns.despine(bottom = True)
#Plotting most popular breeds of Dogss
fig, ax = plt.subplots(figsize = (10,8))
plot =sns.barplot(x ='Count' , y ='Primary_Breed' , data = grouped_by_species_dogs.sort_values('Count',ascending = False).head(10))
plt.ylabel('')
plt.xlabel('')
ax.set_xticks([])
sns.despine(bottom = True)