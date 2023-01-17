import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
car_data = pd.read_csv('../input/datasets3.csv')
car_data.shape
car_data.head()
pd.value_counts(car_data['Fuel Type'])
car_data.loc[car_data['Fuel Type'] == '\xa0 Diesel \xa0', 'Fuel Type'] = 'Diesel'

car_data.loc[car_data['Fuel Type'] == '\xa0 Petrol \xa0', 'Fuel Type'] = 'Petrol'

car_data.loc[car_data['Fuel Type'] == 'petrol', 'Fuel Type'] = 'Petrol'

car_data.loc[car_data['Fuel Type'] == 'diesel', 'Fuel Type'] = 'Diesel'

car_data.loc[car_data['Fuel Type'] == '\xa0 LPG \xa0', 'Fuel Type'] = 'LPG'

car_data.loc[car_data['Fuel Type'] == '\xa0 CNG \xa0', 'Fuel Type'] = 'CNG'

car_data.loc[car_data['Fuel Type'] == '\xa0 Hybrid \xa0', 'Fuel Type'] = 'Hybrid'

car_data.loc[car_data['Fuel Type'] == '\xa0 CNGPetrol \xa0', 'Fuel Type'] = 'CNGPetrol'

car_data.loc[car_data['Fuel Type'] == '\xa0  \xa0', 'Fuel Type'] = ''
sns.set_style('ticks')

fig, ax = plt.subplots()

fig.set_size_inches(20, 10)



sns.countplot(x="Fuel Type", data=car_data, ax=ax)

sns.despine()
print('The list of Make of the cars are ',car_data['Make'].unique())

print('The total count are ',len(car_data['Make'].unique()))
car_data.loc[car_data['Make'] == 'Maruti ', 'Make'] = 'Maruti'
print('The list of Make of the cars are ',car_data['Make'].unique())

print('The total count are ',len(car_data['Make'].unique()))
sns.set_style('ticks')

fig, ax = plt.subplots()

fig.set_size_inches(28, 15)



sns.countplot(x="Make", data=car_data, ax=ax)

sns.despine()
sns.set_style('ticks')

fig, ax = plt.subplots()

fig.set_size_inches(11.7, 10.27)



sns.boxplot(data=car_data['Price in INR'], showfliers=True)
sns.set_style('ticks')

fig, ax = plt.subplots()

fig.set_size_inches(11.7, 10.27)



sns.boxplot(data=car_data['Price in INR'], showfliers=False)
print('The list of locations are ',car_data['Location'].unique())

print('The total count are ',len(car_data['Location'].unique()))
car_data['Location'] = car_data['Location'].apply(lambda x: str(x).replace('\xa0','').strip())



print('The list of locations are ',car_data['Location'].unique())

print('The total count are ',len(car_data['Location'].unique()))
sns.set_style('ticks')

fig, ax = plt.subplots()

fig.set_size_inches(28, 80)



sns.countplot(y="Location", data=car_data, ax=ax)

sns.despine()
sns.set_style('ticks')

fig, ax = plt.subplots()

fig.set_size_inches(28, 15)



sns.countplot(x="Manufacturing Year", data=car_data, ax=ax)

sns.despine()