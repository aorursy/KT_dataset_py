



import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # for doing Exploratory data analysis(EDA)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

crime = pd.read_csv('/kaggle/input/crimes-in-boston/crime.csv', encoding = 'unicode_escape') #create a data frame 

offense_codes=pd.read_csv('/kaggle/input/crimes-in-boston/offense_codes.csv', encoding = 'unicode_escape') #create a data frame 
crime.head()
crime.columns
crime.info()
crime_by_district=crime.groupby('DISTRICT')['INCIDENT_NUMBER'].count().sort_values(ascending=False)

crime_by_district
crime_by_district.plot(kind='bar')
crime_by_offenseCodeGroup=crime.groupby('OFFENSE_CODE_GROUP')['INCIDENT_NUMBER'].count()

plt.figure(figsize=(20,10))

crime_by_offenseCodeGroup.plot(kind='bar')

crime_by_year=crime.groupby('YEAR')['INCIDENT_NUMBER'].count()

plt.figure(figsize=(20,10))

crime_by_year.plot(kind='bar' )
crime_by_month=crime.groupby('MONTH')['INCIDENT_NUMBER'].count()

plt.figure(figsize=(20,10))

crime_by_month.plot(kind='bar' )
crime_by_yearmonth=crime.groupby(['YEAR','MONTH'])['INCIDENT_NUMBER'].count()



crime_by_yearmonth.unstack(level=0).plot(kind='bar', subplots=True)
crime_by_week=crime.groupby('DAY_OF_WEEK')['INCIDENT_NUMBER'].count()

plt.figure(figsize=(20,10))

crime_by_week.plot(kind='bar' )
crime_by_hour=crime.groupby('HOUR')['INCIDENT_NUMBER'].count()

plt.figure(figsize=(20,10))

crime_by_hour.plot(kind='bar' )
crime.columns

crime.groupby('STREET')['INCIDENT_NUMBER'].count().sort_values(ascending= False).iloc[:10].plot(kind='barh')
crime[(crime['STREET'] == 'WASHINGTON ST' )].groupby('HOUR')['INCIDENT_NUMBER'].count().plot(kind='barh')
crime[(crime['STREET'] == 'BLUE HILL AVE' )].groupby('HOUR')['INCIDENT_NUMBER'].count().plot(kind='barh')
crime['SHOOTING'].value_counts()
crime[crime['SHOOTING'] == 'Y'].groupby('YEAR')['INCIDENT_NUMBER'].count().plot(kind='bar')
crime[crime['SHOOTING'] == 'Y'].groupby('STREET')['INCIDENT_NUMBER'].count().sort_values(ascending=False).iloc[:10].plot(kind='bar')