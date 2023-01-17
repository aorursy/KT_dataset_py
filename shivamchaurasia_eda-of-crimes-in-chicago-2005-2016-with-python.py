import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

plt.style.use('seaborn')
Crime1 = pd.read_csv('../input/Chicago_Crimes_2005_to_2007.csv',error_bad_lines=False)

Crime2 = pd.read_csv('../input/Chicago_Crimes_2008_to_2011.csv',error_bad_lines=False)

Crime3 = pd.read_csv('../input/Chicago_Crimes_2012_to_2017.csv',error_bad_lines=False)



Crimes = pd.concat([Crime1, Crime2, Crime3], ignore_index=False, axis=0)



del Crime1

del Crime2

del Crime3



print('Dataset ready..')



print('Dataset Shape before drop_duplicate : ', Crimes.shape)

Crimes.drop_duplicates(subset=['ID', 'Case Number'], inplace=True)

print('Dataset Shape after drop_duplicate: ', Crimes.shape)
Crimes.drop(['Unnamed: 0', 'Case Number', 'IUCR','Updated On','Year', 'FBI Code', 'Beat','Ward','Community Area', 'Location'], inplace=True, axis=1)
#Let's have a look at the first 5 rows of the dataframe 'Crimes'



Crimes.head(5)
# converting dates to pandas datetime format

Crimes.Date = pd.to_datetime(Crimes.Date, format='%m/%d/%Y %I:%M:%S %p')





# setting the index to be the date will help us a lot later on

Crimes.index = pd.DatetimeIndex(Crimes.Date)
Crimes.shape
Crimes.info()
loc_to_change  = list(Crimes['Location Description'].value_counts()[20:].index)

desc_to_change = list(Crimes['Description'].value_counts()[20:].index)

type_to_change = list(Crimes['Primary Type'].value_counts()[20:].index)



Crimes.loc[Crimes['Location Description'].isin(loc_to_change) , Crimes.columns=='Location Description'] = 'OTHER'

Crimes.loc[Crimes['Description'].isin(desc_to_change) , Crimes.columns=='Description'] = 'OTHER'

Crimes.loc[Crimes['Primary Type'].isin(type_to_change) , Crimes.columns=='Primary Type'] = 'OTHER'
Crimes['Primary Type']         = pd.Categorical(Crimes['Primary Type'])

Crimes['Location Description'] = pd.Categorical(Crimes['Location Description'])

Crimes['Description']          = pd.Categorical(Crimes['Description'])
plt.figure(figsize=(11,5))

Crimes.resample('M').size().plot(legend = False)

plt.title('Number of Crimes per month (2005 - 2016)')

plt.xlabel('Months')

plt.ylabel('Number of Crimes')

plt.show()
plt.figure(figsize=(11,6))

Crimes.resample('D').size().rolling(365).sum().plot()

plt.title('Rolling sum of all Crimes from 2005 - 2016')

plt.ylabel('Number of Crimes')

plt.xlabel('Days')

plt.show()
Crimes_count_date = Crimes.pivot_table('ID', aggfunc=np.size, columns='Primary Type', index=Crimes.index.date, fill_value=0)

Crimes_count_date.index = pd.DatetimeIndex(Crimes_count_date.index)

plo = Crimes_count_date.rolling(365).sum().plot(figsize=(12, 30), subplots=True, layout=(-1, 3), sharex=False, sharey=False)
days = ['Monday','Tuesday','Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

Crimes.groupby([Crimes.index.dayofweek]).size().plot(kind='barh')

plt.ylabel('Days of the week')

plt.yticks(np.arange(7), days)

plt.xlabel('Number of Crimes')

plt.title('Number of Crimes by day of the week')

plt.show()
Crimes.groupby([Crimes.index.month]).size().plot(kind='barh')

plt.ylabel('Months of the year')

plt.xlabel('Number of Crimes')

plt.title('Number of Crimes by month of the year')

plt.show()
plt.figure(figsize=(8,10))

Crimes.groupby([Crimes['Primary Type']]).size().sort_values(ascending=True).plot(kind='barh')

plt.title('Number of Crimes by type')

plt.ylabel('Crime Type')

plt.xlabel('Number of Crimes')

plt.show()
plt.figure(figsize=(8,10))

Crimes.groupby([Crimes['Location Description']]).size().sort_values(ascending=True).plot(kind='barh')

plt.title('Number of Crimes by Location')

plt.ylabel('Crime Location')

plt.xlabel('Number of Crimes')

plt.show()