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
# reading data from csv file using pandas

df = pd.read_csv('/kaggle/input/police_traffic.csv')

df.head()
# dropping unnessary 'Unnamed: 0' column

df.drop(columns=['Unnamed: 0'], inplace=True)

df.info()
# Shape of dataset before cleaning is performed

df.shape
import seaborn as sns

import matplotlib.pyplot as plt



sns.set_style('dark')



%matplotlib inline
# The data entries seem to be missing a few values from a couple of the features

# Show null values

print('Missing Values: \n', df.isnull().sum())



# Since we only analyzing one state, examine feature for abnormalities.

# 'state' and 'county_name' provide no value for analysis so drop them

print('\nAny odd states: \n', df['state'].value_counts())

df.drop(columns=['state', 'county_name'], inplace=True)
# 'driver_gender' is crucial for analysis, thus drop columns to avoid bias

df.dropna(subset=['driver_gender'], inplace=True)

print(df.shape)

print(df.isnull().sum())
# 'search_type' shows some missing values, indicating that no search was carried out on the driver.

df['search_type'].value_counts()
# Show data types of each column

df.dtypes
# View head of dataframe to ensure the data type to be implemented

df.head(2)
# 'search_conducted' and 'is_arrested' are object data types that must be converted to booleans

df[['search_conducted', 'is_arrested']] = df[['search_conducted', 'is_arrested']].astype('bool')

df.dtypes
# 'stop_date' and 'stop_time' must be combined and then converted to DateTimeIndex

df['date_time'] = df['stop_date'].str.cat(df['stop_time'], sep=' ')



# converting to datetime index

df['date_time'] = pd.to_datetime(df['date_time'])



# setting new column 'date_time' to the index column

df.set_index('date_time', inplace=True)

df.head(2)
# shape of dataset

print(df.shape)

# export dataset in csv format

df.to_csv('traffic_stop_data.csv')
# Most common stops

outcomes = df['stop_outcome'].value_counts(normalize=True) * 100

print(outcomes)

outcomes.plot(kind='bar', 

             figsize=(10,5),

             rot=0,

             title='Outcomes of traffic stops');
# ethnicity of drivers

race_outcome = df['driver_race'].value_counts(normalize=True) * 100

print(race_outcome)

# plotting race outcomes

race_outcome.plot(kind='bar', 

                  figsize=(10,5), 

                  rot=0,

                 title='Ethnicty of drivers');
# Gender of drivers

gender_outcome = df['driver_gender'].value_counts(normalize=True) * 100

print(gender_outcome)



# plotting outcomes

gender_outcome.plot(kind='bar',

                   figsize=(10,5),

                   rot=0,

                   title='Gender of drivers');
# white drivers

white = df[df['driver_race']=='White']

# counting perentages of outcomes

white_outcome = white['stop_outcome'].value_counts(normalize=True) * 100

print(white_outcome)



# plotting outcomes

white_outcome.plot(kind='bar',

                  figsize=(10, 5),

                  rot=0,

                  title='Stop outcomes on white drivers');
# violations commited by white race group

white_violation = white['violation'].value_counts(normalize=True) * 100

print(white_violation)



# plotting

white_violation.plot(kind='bar',

                    figsize=(10,5),

                    rot=0,

                    title='Violations done by white drivers');
# black drivers

black = df[df['driver_race']=='Black']

black_outcome = black['stop_outcome'].value_counts(normalize=True)*100

print(black_outcome)



# plotting outcomes

black_outcome.plot(kind='bar',

                  figsize=(10,5),

                  rot=0,

                  title='Stop outcomes on black drivers');
# violations by black drivers

black_violation = black['violation'].value_counts(normalize=True) * 100

print(black_violation)



# plotting

black_violation.plot(kind='bar',

                    figsize=(10,5),

                    rot=0,

                    title='Violations committed by black drivers');
# Hispanic drivers

hispanic = df[df['driver_race']=='Hispanic']

hispanic_outcomes = hispanic['stop_outcome'].value_counts(normalize=True) * 100

print(hispanic_outcomes)



# plotting outcomes

hispanic_outcomes.plot(kind='bar',

                      figsize=(10,5),

                      rot=0,

                      title='Stop outcomes on hispanic drivers');
# violations commited by hispanic race group

hispanic_violation = hispanic['violation'].value_counts(normalize=True) * 100

print(hispanic_violation)



# plotting

hispanic_violation.plot(kind='bar',

                    figsize=(10,5),

                    rot=0,

                    title='Violations committed by hispanic drivers');
data_race_violation = pd.DataFrame({'White': white_violation, 

                         'Black': black_violation, 

                         'Hispanic': hispanic_violation,}, index=None).sort_values(by='White',

                                                                                  ascending=False)

print('Violations:')

data_race_violation
data_race_outcomes = pd.DataFrame({'White': white_outcome,

                                  'Black': black_outcome,

                                  'Hispanic': hispanic_outcomes}, 

                                  index=None).sort_values(by='White', 

                                                          ascending=False)



print('Outcomes:')

data_race_outcomes
# Male gender

males = df[df['driver_gender']=='M']

males_outcomes = males['stop_outcome'].value_counts(normalize=True) * 100

print(males_outcomes)



# plotting outcomes

males_outcomes.plot(kind='bar',

                   figsize=(10,5),

                   rot=0,

                   title='Stop outcomes of male drivers');
# violations commited by male gender group

male_violation = males['violation'].value_counts(normalize=True) * 100

print(male_violation)



# plotting

male_violation.plot(kind='bar',

                    figsize=(10,5),

                    rot=0,

                   title='Violations committed by male drivers');
# Female outcomes

females = df[df['driver_gender']=='F']

females_outcomes = females['stop_outcome'].value_counts(normalize=True) * 100

print(females_outcomes)



# plotting outcomes

females_outcomes.plot(kind='bar',

                     figsize=(10,5),

                     rot=0,

                     title='Stop outcomes of female drivers');
# violations commited by female gender group

female_violation = females['violation'].value_counts(normalize=True) * 100

print(female_violation)



# plotting

female_violation.plot(kind='bar',

                    figsize=(10,5),

                    rot=0,

                     title='Violations committed by female divers');
data_gender = pd.DataFrame({'Females': female_violation,

                           'Males': male_violation}, index=None).sort_values(by='Females', 

                                                                             ascending=False)

data_gender
# dataframe 'white' from previous analysis

white_search_rate = white['search_conducted'].value_counts(normalize=True) * 100

white_search_rate
black_search_rate = black['search_conducted'].value_counts(normalize=True) * 100

black_search_rate
hisp_search_rate = hispanic['search_conducted'].value_counts(normalize=True) * 100

hisp_search_rate
# using the 'males' and 'females' dataframe from the previous analysis

male_search = males['search_conducted'].value_counts(normalize=True) * 100

male_search
female_search = females['search_conducted'].value_counts(normalize=True) * 100

female_search
total_frisks = df['search_type'].str.contains('Protective Frisk', na=False).sum()

total_frisks
df_male_frisk = males[males['search_conducted'] == True]

male_frisk = df_male_frisk['search_type'].str.contains('Protective Frisk').sum()

male_frisk
df_female_frisk = females[females['search_conducted']==True]

female_frisk = df_female_frisk['search_type'].str.contains('Protective Frisk').sum()

female_frisk
male_frisk_rate = round(male_frisk/total_frisks * 100, 3)

female_frisk_rate = round(female_frisk/total_frisks * 100, 3)

print(f'Male Frisk Rate: {male_frisk_rate}%')

print(f'Female Frisk Rate: {female_frisk_rate}%')
# drug-related stops

annual_drug_stops = df['drugs_related_stop'].resample('A').mean()

annual_drug_stops.plot(figsize=(10,5),

                      title='Mean of annual drug related stops');
