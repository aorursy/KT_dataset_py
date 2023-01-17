import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
crime = pd.read_csv('../input/crime-classifcication/Crime1.csv',usecols=['Dates','Category','Descript','DayOfWeek','PdDistrict','Resolution','Address'])
crime.head()
crime.info()
crime.dtypes
crime.describe()
crime['Dates'] = pd.to_datetime(crime['Dates'])
crime.dtypes
crime['Category'].value_counts()
crime['DayOfWeek'].value_counts()
crime['PdDistrict'].value_counts()
crime['Resolution'].value_counts()
crime_category = crime.groupby('Category')['Category'].count().sort_values(ascending=False)
crime_category
plt.figure(figsize=(10,8))

crime_category.plot(kind='barh')

plt.xlabel('Count')

plt.title('Number of times each Crime Category took place')

plt.show()
print('We can see from the above plot that LARCENY/THEFT is the most common crime category')
crime.head()
plt.figure(figsize=(10,8))

sns.countplot(crime['DayOfWeek'])

plt.title('Day of the week on which most crimes take place')

plt.show()
print('On SATURDAY most crimes take place')
plt.figure(figsize=(12,8))

sns.countplot(crime['PdDistrict'])

plt.show()
print('SOUTHERN district is famous in terms of crimes')
larseny_descript = crime.loc[crime['Category']=='LARCENY/THEFT','Descript'].value_counts()
larseny_descript
# grouping the data set on the basis of 'Category' and 'Resolution' columns

category_resolution = crime.groupby(['Category','Resolution'])['Category'].count()
category_resolution
# filtering out 'LARCENY/THEFT' cases

larceny_cases = category_resolution['LARCENY/THEFT']
larceny_cases
larceny_cases.plot(kind='bar')

plt.show()
print('There was no resolution for majority of LARCENY/THEFT cases')
# grouping the data based on 'Category' and 'DayOfWeek' columns

category_day = crime.groupby(['Category','DayOfWeek'])['Category'].count()
category_day
# filtering out 'LARCENY/THEFT' cases

larceny_day = category_day['LARCENY/THEFT']
larceny_day
plt.figure(figsize=(10,8))

larceny_day.plot(kind='bar')

plt.ylabel('Count')

plt.show()
larseny_address = crime.groupby(['Category','PdDistrict','Address'])['Address'].count()['LARCENY/THEFT']['SOUTHERN'].sort_values(ascending=False)
larseny_address
print('No specific address')