import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt



source_metadata = pd.read_csv('../input/crime.csv')

source_metadata.head()
df = pd.DataFrame([source_metadata['YEAR'], source_metadata['TYPE']]).T

df.head()
dataTotal = df.groupby(['YEAR']).count().reset_index()

dataTotal.columns = ['Year','Total']

dataTotal
plt.plot(dataTotal['Year'], dataTotal['Total'])

plt.title('Total No. of Occurences Crime in Vancouver per Year')

plt.xlabel('Year')

plt.ylabel('Total No. of Occurences Crime')

plt.show()
df2 = pd.DataFrame([source_metadata['YEAR'], source_metadata['MONTH'], source_metadata['TYPE']]).T

df2.head()
df2.sort_values(by=['MONTH'])
dataAveMon = df2.groupby(['YEAR','MONTH']).count().reset_index()

dataAveMon.head()
dataAveMon.drop('YEAR', axis = 1)
dataAveMon = dataAveMon.groupby(['MONTH'])['TYPE'].mean().reset_index()
dataAveMon.columns = ['Month','Ave']

dataAveMon
plt.plot(dataAveMon['Month'], dataAveMon['Ave'])

plt.title('Average No. of Occurences Crime in Vancouver per Month')

plt.xlabel('Month')

plt.ylabel('Average No. of Occurences Crime')

plt.show()
df3 = pd.DataFrame(source_metadata['TYPE'])

df3.head()
dataCrime = df3['TYPE'].value_counts()

dataCrime
c = dataCrime.plot(kind = 'bar')
df5 = pd.DataFrame([source_metadata['YEAR'], source_metadata['NEIGHBOURHOOD'], source_metadata['TYPE']]).T
df5.head()
dataAveCrime = df5.groupby(['YEAR','NEIGHBOURHOOD']).count().reset_index()

dataAveCrime.head()
dataAveCrime = dataAveCrime.drop('YEAR', axis = 1)
dataAveCrime.columns = ["Neighborhood", "Ave"]
dataAveCrime = dataAveCrime.groupby(['Neighborhood'])['Ave'].mean()
dataAveCrime.head()
cr = dataAveCrime.plot(kind = 'bar')