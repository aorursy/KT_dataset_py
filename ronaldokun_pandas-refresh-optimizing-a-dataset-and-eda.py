# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pandas_profiling

import matplotlib.pyplot as plt

from pprint import pprint as pp



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

%matplotlib inline
INPUT = '/kaggle/input/stanford-open-policing-project/police_project.csv'





# Use 3 decimal places in output display

pd.set_option("display.precision", 3)



# Don't wrap repr(DataFrame) across additional lines

pd.set_option("display.expand_frame_repr", False)



# Set max rows displayed in output to 25

pd.set_option("display.max_rows", 25)



df = pd.read_csv(INPUT)



df.head()
df.info()
df.profile_report(style={'full_width':True})
df.shape
df.isnull().sum()
df.drop('county_name', axis='columns', inplace=True)
df.dropna(subset=['driver_gender'], inplace=True)

df.isnull().sum()
df.loc[df.driver_age.isnull(), ['driver_age']] = int(df.driver_age.mean())

df.loc[df.driver_age_raw.isnull(), ['driver_age_raw']] = int(df.driver_age_raw.mean())
df.info()
# When assigning to columns, only the square brackets notation works.

df['is_arrested'] = df.is_arrested.astype('bool')

df.info()
df['driver_age'] = df['driver_age'].astype('int8')

df['driver_age_raw'] = df['driver_age_raw'].astype('int16')

df.info()
for col in ['driver_gender', 'driver_race', 'violation_raw', 'violation', 'stop_outcome']:

    df[col] = df[col].astype('category')

    

df.info()
datetime = df.stop_date.str.cat(df.stop_time, sep= ' ')

df['stop_datetime'] = pd.to_datetime(datetime)
df.set_index('stop_datetime', inplace=True)

df.head()
# Now we can drop the redundant columns

df.drop(['stop_date', 'stop_time'], axis='columns', inplace=True)

df.head()
df.stop_outcome.value_counts()
# Percentage from the total

df.stop_outcome.value_counts() / df.shape[0]
# As usual there is a method which do that for us

df.stop_outcome.value_counts(normalize=True)
df.driver_race.value_counts()
male = df[df.driver_gender == 'M']

female = df[df.driver_gender == 'F']



print("Female Violations")

pp(female.violation.value_counts(normalize=True))



print("\nMale Violations")

pp(male.violation.value_counts(normalize=True))

print(f"Female Records: {female.shape[0]}\nMale Records: {male.shape[0]}")
arrested_females = df[(df.driver_gender == 'F') & (df.is_arrested == True)]

arrested_males = df[(df.driver_gender == 'M') & (df.is_arrested == True)]
print(f"Arrested Females: {arrested_females.shape[0]}\nArrested Males: {arrested_males.shape[0]}")
female_and_speeding = df[(df.driver_gender == 'F') & (df.violation == 'Speeding')]

male_and_speeding = df[(df.driver_gender == 'M') & (df.violation == 'Speeding')]



print("Female Outcomes After Speeding")

print(female_and_speeding.stop_outcome.value_counts(normalize=True))

print("\nMale Outcomes After Speeding")

print(male_and_speeding.stop_outcome.value_counts(normalize=True))
print(df.search_conducted.dtype)

print(df.search_conducted.value_counts(normalize=True))



print("\nPercentage of Searched Vehicles:")

print(f'{df.search_conducted.mean() * 100:.2f}%')
print("\nPercentage of Searched Vehicles (Female):")



print(f"{df[df.driver_gender == 'F'].search_conducted.mean() * 100:.2f}%")



print("\nPercentage of Searched Vehicles (Male):")



print(f"{df[df.driver_gender == 'M'].search_conducted.mean() * 100:.2f}%")
print("\nSearched Vehicles by gender:")



df.groupby('driver_gender').search_conducted.mean()
print(df.groupby(['violation', 'driver_gender']).search_conducted.mean())
print(df.search_type.value_counts())



# Check if 'search_type' contains the string 'Protective Frisk'

df['frisk'] = df.search_type.str.contains('Protective Frisk', na=False)



# Take the sum of 'frisk'

print(df.frisk.sum())
searched = df[df.search_conducted == True]



print(searched.frisk.mean())



# Calculate the frisk rate for each gender

print(searched.groupby('driver_gender').frisk.mean())
print(f"Mean of Arrests: {df.is_arrested.mean():.3f}")

print("Hourly Arrest Rates")

hourly_arrest_rate = df.groupby(df.index.hour).is_arrested.mean()

pp(hourly_arrest_rate)
hourly_arrest_rate.plot()



# Add the xlabel, ylabel, and title

plt.xlabel('Hour')

plt.ylabel('Arrest Rate')

plt.title('Arrest Rate by Time of Day')



# Display the plot

plt.show()
annual_drug_rate = df.drugs_related_stop.resample("A").mean()



annual_drug_rate.plot()

plt.xlabel('Year')

plt.ylabel('Drug Found Rate')

plt.title('Yearly Drug Related Stops')

plt.show()
annual_search_rate = df.search_conducted.resample('A').mean()



annual = pd.concat([annual_drug_rate, annual_search_rate], axis=1)



annual.plot(subplots=True)

plt.xlabel('Year')

plt.ylabel('Annual Rate')

plt.title('Yearly Searchs and Drug Related Stops')

plt.show()
table = pd.crosstab(df.driver_race, df.violation)

table
table = pd.crosstab(df.driver_race, df.violation, normalize=True)

table
table.plot(kind='barh')

plt.show()
print(df.stop_duration.unique())



# Create a dictionary that maps strings to integers

mapping = {'0-15 Min': 8, '16-30 Min': 23, '30+ Min': 45}



# Convert the 'stop_duration' strings to integers using the 'mapping'

df['stop_minutes'] =df.stop_duration.map(mapping)

print(df.stop_minutes.unique())
df.dropna(subset=['stop_minutes'], inplace=True)
stop_length = df.groupby('violation_raw').stop_minutes.mean()

stop_length.sort_values().plot(kind='barh')



plt.xlabel('Approximate Duration in Minutes')

plt.ylabel('Detailed Violation')

plt.title("Stopping Duration by Violation")

plt.show()