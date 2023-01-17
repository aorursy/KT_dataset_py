import numpy as np 
import pandas as pd
import csv
import datetime
import matplotlib as mpl
import seaborn as sns
import matplotlib.pyplot as plt
from ggplot import *
from  matplotlib.ticker import FuncFormatter

data = pd.read_csv('../input/Health_AnimalBites.csv')

data.sample(5)
data.shape
data.columns.values
missing_values = data.isnull().sum()
missing_values
sns.countplot(data['SpeciesIDDesc'])
plt.title("Animals involved in bites", fontsize=20)
plt.xlabel('Animal species', fontsize=18)
plt.ylabel('Bites', fontsize=18)
plt.xticks(fontsize=12, rotation=90)
plt.show()
dog_breeds = data.where(data['SpeciesIDDesc'] == "DOG")

plt.figure(figsize=(18,6))
sns.countplot(dog_breeds['BreedIDDesc'])
plt.title("Dog breeds involved in bites", fontsize=20)
plt.xlabel('Dog breeds', fontsize=18)
plt.ylabel('Bites', fontsize=18)
plt.xticks(fontsize=10, rotation=90)
plt.show()
plt.figure(figsize=(14,4))
sns.countplot(dog_breeds['BreedIDDesc'], order = dog_breeds['BreedIDDesc'].value_counts().iloc[0:30].index)
plt.title("Dog breeds involved in bites (30 most common)", fontsize=20)
plt.xlabel('Dog breeds', fontsize=18)
plt.ylabel('Bites', fontsize=18)
plt.xticks(fontsize=12, rotation=90)
plt.show()
dog_breeds['BreedIDDesc'].value_counts().iloc[0:30]
plt.figure(figsize=(6,4))
sns.countplot(dog_breeds['GenderIDDesc'])
plt.title("Gender of dogs involved in bites", fontsize=20)
plt.xlabel('Genders', fontsize=18)
plt.ylabel('Bites', fontsize=18)
plt.xticks(fontsize=14)
plt.show()
plt.figure(figsize=(16,4))
sns.countplot(dog_breeds['BreedIDDesc'].where(dog_breeds['GenderIDDesc']=='FEMALE'), order = dog_breeds['BreedIDDesc'].value_counts().iloc[0:30].index)
plt.title("Females involved in bites", fontsize=20)
plt.xlabel('Breeds', fontsize=18)
plt.ylabel('Bites', fontsize=18)
plt.xticks(fontsize=14, rotation=90)
plt.show()
plt.figure(figsize=(16,4))
sns.countplot(dog_breeds['BreedIDDesc'].where(dog_breeds['GenderIDDesc']=='MALE'), order = dog_breeds['BreedIDDesc'].value_counts().iloc[0:30].index)
plt.title("Males involved in bites", fontsize=20)
plt.xlabel('Breeds', fontsize=18)
plt.ylabel('Bites', fontsize=18)
plt.xticks(fontsize=14, rotation=90)
plt.show()
plt.figure(figsize=(16,4))
sns.countplot(dog_breeds['BreedIDDesc'].where(dog_breeds['GenderIDDesc']=='UNKNOWN'), order = dog_breeds['BreedIDDesc'].value_counts().iloc[0:30].index)
plt.title("Unknown gender dogs involved in bites", fontsize=20)
plt.xlabel('Breeds', fontsize=18)
plt.ylabel('Bites', fontsize=18)
plt.xticks(fontsize=14, rotation=90)
plt.show()
# Trying to parse the dates I found that there was a date set as 5013-07-15 00:00:00, which is possibly a typo for 2013.
#dog_breeds[dog_breeds.bite_date == '5013-07-15 00:00:00'] #index = 4490
#Considering it is the middle of the 2013 data, I am probably right, so I will fix it

#no warning for changing values
pd.options.mode.chained_assignment = None 

dog_breeds.bite_date[4490] = '2013-07-15 00:00:00'
dog_breeds['bite_date'] = pd.to_datetime(dog_breeds['bite_date'].dropna(), format = "%Y/%m/%d %H:%M:%S")
year_bites = dog_breeds['bite_date'].dt.year

plt.figure(figsize=(16,4))
sns.countplot(year_bites)
plt.title("Dog bites per year", fontsize=20)
plt.xlabel('Year', fontsize=18)
plt.ylabel('Bites', fontsize=18)
plt.xticks(fontsize=14, rotation=90)
plt.show()
month_bites = dog_breeds['bite_date'].dt.month

plt.figure(figsize=(10,4))
sns.countplot(month_bites)
plt.title("Dog bites per month", fontsize=20)
plt.xlabel('Month', fontsize=18)
plt.ylabel('Bites', fontsize=18)
plt.xticks(fontsize=14, rotation=90)
plt.show()
day_bites = dog_breeds['bite_date'].dt.day

plt.figure(figsize=(16,4))
sns.countplot(day_bites)
plt.title("Dog bites per day of the month", fontsize=20)
plt.xlabel('Day', fontsize=18)
plt.ylabel('Bites', fontsize=18)
plt.xticks(fontsize=14, rotation=90)
plt.show()
plt.figure(figsize=(6,4))
sns.countplot(dog_breeds['ResultsIDDesc'])
plt.title("Results of rabies exam for dogs involved in bites", fontsize=20)
plt.xlabel('Breeds', fontsize=18)
plt.ylabel('Bites', fontsize=18)
plt.xticks(fontsize=14)
plt.show()
print("Rabies test results after dog bites")
dog_breeds['ResultsIDDesc'].value_counts()
print("Where were people bitten by dogs")
print(dog_breeds['WhereBittenIDDesc'].value_counts())
plt.figure(figsize=(16,4))
sns.countplot(dog_breeds['BreedIDDesc'].where(dog_breeds['WhereBittenIDDesc']=='BODY'), order = dog_breeds['BreedIDDesc'].value_counts().iloc[0:30].index)
plt.title("Dog bites in the body per breed", fontsize=20)
plt.xlabel('Breeds', fontsize=18)
plt.ylabel('Bites', fontsize=18)
plt.xticks(fontsize=14, rotation=90)
plt.show()
plt.figure(figsize=(16,4))
sns.countplot(dog_breeds['BreedIDDesc'].where(dog_breeds['WhereBittenIDDesc']=='HEAD'), order = dog_breeds['BreedIDDesc'].value_counts().iloc[0:30].index)
plt.title("Dog bites in the body per breed", fontsize=20)
plt.xlabel('Breeds', fontsize=18)
plt.ylabel('Bites', fontsize=18)
plt.xticks(fontsize=14, rotation=90)
plt.show()
body_bites = dog_breeds['BreedIDDesc'].where(dog_breeds['WhereBittenIDDesc']=='BODY').value_counts()
head_bites = dog_breeds['BreedIDDesc'].where(dog_breeds['WhereBittenIDDesc']=='HEAD').value_counts()

dogs_ratio = head_bites / body_bites
dogs_ratio = dogs_ratio.sort_values(ascending=False).iloc[0:30]

print("Ratio of bites on head over bites on body - 30 highest")
print(dogs_ratio)
plt.figure(figsize=(10,4))
sns.countplot(dog_breeds['bite_date'].dt.year.where(dog_breeds['BreedIDDesc']=='PIT BULL'))
plt.title("Pit bull bites over the years", fontsize=20)
plt.xlabel('Year', fontsize=18)
plt.ylabel('Bites', fontsize=18)
plt.xticks(fontsize=14, rotation=90)
plt.show()
cat_breeds = data.where(data['SpeciesIDDesc'] == "CAT")

plt.figure(figsize=(6,4))
sns.countplot(cat_breeds['GenderIDDesc'])
plt.title("Gender of cats involved in bites", fontsize=20)
plt.xlabel('Genders', fontsize=18)
plt.ylabel('Bites', fontsize=18)
plt.xticks(fontsize=14)
plt.show()
#parsing dates so we can look at the time distribution on cat bites
cat_breeds['bite_date'] = pd.to_datetime(cat_breeds['bite_date'].dropna(), format = "%Y/%m/%d %H:%M:%S")
year_cat = cat_breeds['bite_date'].dt.year

plt.figure(figsize=(10,4))
sns.countplot(year_cat)
plt.title("Cat bites per year", fontsize=20)
plt.xlabel('Year', fontsize=18)
plt.ylabel('Bites', fontsize=18)
plt.xticks(fontsize=14, rotation=90)
plt.show()
month_cat = cat_breeds['bite_date'].dt.month

plt.figure(figsize=(10,4))
sns.countplot(month_cat)
plt.title("Cat bites per month", fontsize=20)
plt.xlabel('Month', fontsize=18)
plt.ylabel('Bites', fontsize=18)
plt.xticks(fontsize=14, rotation=90)
plt.show()
day_cat = cat_breeds['bite_date'].dt.day

plt.figure(figsize=(16,4))
sns.countplot(day_cat)
plt.title("Cat bites per day", fontsize=20)
plt.xlabel('Day of month', fontsize=18)
plt.ylabel('Bites', fontsize=18)
plt.xticks(fontsize=14, rotation=90)
plt.show()
print("Where have people been bitten by cats")
print(dog_breeds['WhereBittenIDDesc'].value_counts())
print("Rabies results for cats involved in bites")
print(dog_breeds['ResultsIDDesc'].value_counts())
others = data.where((data['SpeciesIDDesc'] != "DOG")&(data['SpeciesIDDesc'] != "CAT"))

plt.figure(figsize=(8,4))
sns.countplot(others['SpeciesIDDesc'], order = others['SpeciesIDDesc'].value_counts().index)
plt.title("Other animals involved in bites", fontsize=20)
plt.xlabel('Species', fontsize=18)
plt.ylabel('Bites', fontsize=18)
plt.xticks(fontsize=10, rotation=90)
plt.show()
others['bite_date'] = pd.to_datetime(others['bite_date'].dropna(), format = "%Y/%m/%d %H:%M:%S")
year_others = others['bite_date'].dt.year

plt.figure(figsize=(10,4))
sns.countplot(year_others)
plt.title("Bites per year", fontsize=20)
plt.xlabel('Year', fontsize=18)
plt.ylabel('Bites', fontsize=18)
plt.xticks(fontsize=14, rotation=90)
plt.show()
month_others = others['bite_date'].dt.month

plt.figure(figsize=(10,4))
sns.countplot(month_others)
plt.title("Bites per month", fontsize=20)
plt.xlabel('Month', fontsize=18)
plt.ylabel('Bites', fontsize=18)
plt.xticks(fontsize=14, rotation=90)
plt.show()
day_others = others['bite_date'].dt.day

plt.figure(figsize=(16,4))
sns.countplot(day_others)
plt.title("Bites per day of the month", fontsize=20)
plt.xlabel('Day of month', fontsize=18)
plt.ylabel('Bites', fontsize=18)
plt.xticks(fontsize=14, rotation=90)
plt.show()
print("Where have people been bitten by animals")
print(others['WhereBittenIDDesc'].value_counts())
print("Rabies results for animals involved in bites")
print(others['ResultsIDDesc'].value_counts())