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
offense_codes = pd.read_csv("/kaggle/input/crimes-in-boston/offense_codes.csv", encoding='latin-1')

crime = pd.read_csv("/kaggle/input/crimes-in-boston/crime.csv", encoding='latin-1')
offense_codes.sort_values(by=["CODE"])
print(crime.describe())

crime.head(5)
import matplotlib.pyplot as plt

import seaborn as sns



sns.countplot(crime.OFFENSE_CODE)

plt.show()



sns.countplot(crime.MONTH)

plt.show()



sns.countplot(crime.DAY_OF_WEEK)

plt.show()



sns.countplot(crime.DISTRICT)

plt.show()



sns.countplot(crime.OFFENSE_CODE_GROUP)

plt.show()
crime.groupby(['DISTRICT']).mean()
districts = crime.DISTRICT.unique()

district_crimes = {}

for district in districts:

    district_crime = crime.loc[crime['DISTRICT'] == district]

    district_crimes[district] = district_crime
for district in list(district_crimes.keys()):

    if district == district:  # skip if nan 

        sns.countplot(district_crimes[district].HOUR)

        plt.title(district)

        plt.show()
crime.OFFENSE_CODE_GROUP.unique()
top_crimes = crime.groupby(['OFFENSE_CODE_GROUP']).count().sort_values(['INCIDENT_NUMBER'], ascending=False).head(20)

top_crimes.INCIDENT_NUMBER.plot.bar()
def scatter_incident(incident, color):

    motor_accidents = crime.loc[crime['OFFENSE_CODE_GROUP'] == incident]

    motor_accidents = motor_accidents.dropna(subset=['Lat', 'Long'])

    motor_accidents = motor_accidents[motor_accidents['Lat']!=-1]

    motor_accidents = motor_accidents[motor_accidents['Long']!=-1]

    motor_accidents.plot.scatter(x='Lat',y='Long',s=0.1,c=color)



scatter_incident('Motor Vehicle Accident Response', 'green')

scatter_incident('Auto Theft', 'red')

scatter_incident('Medical Assistance', 'blue')

scatter_incident('Larceny', 'black')

scatter_incident('Investigate Person', 'green')

scatter_incident('Drug Violation', 'red')

scatter_incident('Larceny From Motor Vehicle', 'blue')



true_crimes = ['Larceny', 'Vandalism', 'Auto Theft', 'Robbery', 'Larceny From Motor Vehicle', 'Residential Burglary', 'Simple Assault', 'Ballistics', 'Drug Violation', 'Disorderly Conduct', 'Fraud', 'Aggravated Assault', 'Firearm Violations', 'Arson', 'Bomb Hoax', 'Firearm Discovery',

              'Commercial Burglary', 'HOME INVASION', 'Prostitution', 'Homicide', 'Explosives', 'Criminal Harassment', 'Manslaughter']





true_crime = crime.loc[crime['OFFENSE_CODE_GROUP'].isin(true_crimes)]
plt.figure(figsize=(16, 6))

plt.subplot(121)

sns.countplot(crime.OFFENSE_CODE)



plt.subplot(122)

sns.countplot(true_crime.OFFENSE_CODE)

plt.show()



plt.figure(figsize=(16, 6))

plt.subplot(121)

sns.countplot(crime.MONTH)



plt.subplot(122)

sns.countplot(true_crime.MONTH)

plt.show()



plt.figure(figsize=(16, 6))

plt.subplot(121)

sns.countplot(crime.DAY_OF_WEEK)



plt.subplot(122)

sns.countplot(true_crime.DAY_OF_WEEK)

plt.show()



plt.figure(figsize=(16, 6))

plt.subplot(121)

sns.countplot(crime.DISTRICT, order = crime.DISTRICT.value_counts().index)



plt.subplot(122)

sns.countplot(true_crime.DISTRICT, order = crime.DISTRICT.value_counts().index)

plt.show()



plt.figure(figsize=(16, 6))

plt.subplot(121)

g = sns.countplot(crime.OFFENSE_CODE_GROUP, order = crime.OFFENSE_CODE_GROUP.value_counts().index)

g.set_xticklabels(g.get_xticklabels(), rotation=90)

plt.subplot(122)

g = sns.countplot(true_crime.OFFENSE_CODE_GROUP, order = true_crime.OFFENSE_CODE_GROUP.value_counts().index)

g.set_xticklabels(g.get_xticklabels(), rotation=90)

plt.show()

robberies = ['Larceny', 'Auto Theft', 'Robbery', 'Larceny From Motor Vehicle', 'Residential Burglary', 'Commercial Burglary']

robbery_crimes = crime.loc[crime['OFFENSE_CODE_GROUP'].isin(robberies)]



plt.figure(figsize=(16, 6))

plt.subplot(121)

sns.countplot(crime.OFFENSE_CODE)



plt.subplot(122)

sns.countplot(robbery_crimes.OFFENSE_CODE)

plt.show()



plt.figure(figsize=(16, 6))

plt.subplot(121)

sns.countplot(crime.MONTH)



plt.subplot(122)

sns.countplot(robbery_crimes.MONTH)

plt.show()



plt.figure(figsize=(16, 6))

plt.subplot(121)

sns.countplot(crime.DAY_OF_WEEK)



plt.subplot(122)

sns.countplot(robbery_crimes.DAY_OF_WEEK)

plt.show()



plt.figure(figsize=(16, 6))

plt.subplot(121)

sns.countplot(crime.DISTRICT, order = crime.DISTRICT.value_counts().index)



plt.subplot(122)

sns.countplot(robbery_crimes.DISTRICT, order = crime.DISTRICT.value_counts().index)

plt.show()



plt.figure(figsize=(20, 6))

plt.subplot(121)

g = sns.countplot(crime.OFFENSE_CODE_GROUP, order = crime.OFFENSE_CODE_GROUP.value_counts().index)

g.set_xticklabels(g.get_xticklabels(), rotation=90)

plt.subplot(122)

g = sns.countplot(robbery_crimes.OFFENSE_CODE_GROUP, order = robbery_crimes.OFFENSE_CODE_GROUP.value_counts().index)

g.set_xticklabels(g.get_xticklabels(), rotation=90)

plt.show()