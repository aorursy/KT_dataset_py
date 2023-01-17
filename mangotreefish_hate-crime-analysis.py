import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

import seaborn as sns

df = pd.read_csv('../input/hate-crime/hate_crime.csv')



df.head()
df.dtypes #Data types look good!
df.describe()
print(df['INCIDENT_ID'].size, df['ADULT_VICTIM_COUNT'].size,df['ADULT_OFFENDER_COUNT'].size)
df['ADULT_VICTIM_COUNT'].count()
df.isnull().sum() 
df["OFFENDER_RACE"].fillna('Unknown', inplace=True) #Fill Nulls as unknown
df.isnull().sum() 
df['OFFENDER_RACE'].value_counts()
plt.figure(figsize=(10,5))



df['OFFENDER_RACE'].value_counts().plot(kind='barh')



plt.title('Incidents by Offender Race')

plt.ylabel('Race')

plt.xlabel('Number of Incidents')



plt.show()

df['REGION_NAME'].value_counts()
plt.figure(figsize=(10,5))



df['REGION_NAME'].value_counts().plot(kind='barh')



plt.title('Incidents by Region')

plt.ylabel('Region')

plt.xlabel('Number of Incidents')



plt.show()

plt.figure(figsize=(10,5))



df['STATE_ABBR'].value_counts(ascending=False).nlargest(10).plot(kind='barh')



plt.title('Incidents by State')

plt.ylabel('State')

plt.xlabel('Number of Incidents')



plt.show()
df['LOCATION_NAME'].value_counts().nlargest(10)





plt.figure(figsize=(10,5))



df['LOCATION_NAME'].value_counts(ascending=False).nlargest(10).plot(kind='barh')



plt.title('Top 10 Incident Locations')

plt.ylabel('Location')

plt.xlabel('Number of Incidents')



plt.show()
df['VICTIM_TYPES'].value_counts().nlargest(8)
plt.figure(figsize=(10,5))



df['VICTIM_TYPES'].value_counts().nlargest(8).plot(kind='barh')



plt.title('Incidents by Victim Type')

plt.ylabel('Victim Type')

plt.xlabel('Number of Incidents')



plt.show()
df['OFFENSE_NAME'].value_counts().nlargest(15)
plt.figure(figsize=(8,5))



df['OFFENSE_NAME'].value_counts().nlargest(15).plot(kind='barh')



plt.title('Incidents by Type of Offense ')

plt.ylabel('Offense Name')

plt.xlabel('Number of Incidents')



plt.show()
df['STATE_ABBR'].value_counts().nlargest(10)
year = df.groupby(by='DATA_YEAR').count().reset_index() #Group data by year
year
year_data =year[['DATA_YEAR','INCIDENT_ID']] #Make a new data frame to 
year_data
years=list(map(str,range(1991,2018))) #Create the graph



plt.style.use('seaborn')

year_data.plot.line(x='DATA_YEAR')



plt.title('Number of Incidents Per Year')



plt.xlabel('Year')

plt.ylabel('Number of Incidents')
df['BIAS_DESC'].value_counts().nlargest(20) #Get the 20 largest
plt.figure(figsize=(10,5)) #Plot it



df['BIAS_DESC'].value_counts(ascending=False).nlargest(20).plot(kind='barh')



plt.title('Top 20 Bias Discriminations')

plt.ylabel('Type of Discrimination')

plt.xlabel('Number of Incidents')



plt.show()