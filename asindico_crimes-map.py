# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
crimes1 = pd.read_csv('../input/Chicago_Crimes_2005_to_2007.csv',error_bad_lines=False)

crimes2 = pd.read_csv('../input/Chicago_Crimes_2008_to_2011.csv',error_bad_lines=False)

crimes3 = pd.read_csv('../input/Chicago_Crimes_2012_to_2017.csv',error_bad_lines=False)

crimes = pd.concat([crimes1, crimes2, crimes3], ignore_index=False, axis=0)

crimes.head()
crimes['Primary Type'].unique()	
crimes.columns.get_loc('Latitude')
minLat = crimes['Latitude'].mean() - crimes['Latitude'].std()

maxLat = crimes['Latitude'].mean() + crimes['Latitude'].std()

minLon = crimes['Longitude'].mean() - crimes['Longitude'].std()

maxLon = crimes['Longitude'].mean() + crimes['Longitude'].std()

ndf = crimes[crimes['Latitude'] <= maxLat]

ndf = ndf[ndf['Latitude'] >= minLat]

ndf  = ndf[ndf['Longitude']<=maxLon]

ndf  = ndf[ndf['Longitude']>minLon]



import matplotlib.pyplot as plt

import seaborn as sns



n3df = ndf[ndf['Year']>=2005]

nd3df = ndf[ndf['Year']<=2007]

nnhdf = n3df[(n3df['Primary Type']!= 'HOMICIDE') & 

            (n3df['Primary Type']!=  'THEFT') &

            (n3df['Primary Type']!= 'ROBBERY')]

nhdf = n3df[n3df['Primary Type']=='HOMICIDE']

ntdf = n3df[(n3df['Primary Type']=='THEFT')| 

           (n3df['Primary Type']=='ROBBERY')]



f,ax= plt.subplots(1,1,figsize=(12,10))

plt.xlim(41.72,42)

other = ax.scatter(nnhdf['Latitude'],nnhdf['Longitude'], s=0.4, alpha=1, label = 'other crimes')

homicides = ax.scatter(nhdf['Latitude'],nhdf['Longitude'], s=5, alpha=1,color='red', label='homicides')

tr = ax.scatter(ntdf['Latitude'],ntdf['Longitude'], s=0.3, alpha=0.4,color='yellow',label='thefts and robberies')

plt.title('Crimes Map - From 2005 to 2007')

lgnd = plt.legend(handles=[homicides, tr,other])
import matplotlib.pyplot as plt

import seaborn as sns



n3df = ndf[ndf['Year']>=2008]

nd3df = ndf[ndf['Year']<=2011]

nnhdf = n3df[(n3df['Primary Type']!= 'HOMICIDE') & 

            (n3df['Primary Type']!=  'THEFT') &

            (n3df['Primary Type']!= 'ROBBERY')]

nhdf = n3df[n3df['Primary Type']=='HOMICIDE']

ntdf = n3df[(n3df['Primary Type']=='THEFT')| 

           (n3df['Primary Type']=='ROBBERY')]



f,ax= plt.subplots(1,1,figsize=(12,10))

plt.xlim(41.72,42)

other = ax.scatter(nnhdf['Latitude'],nnhdf['Longitude'], s=0.4, alpha=1, label = 'other crimes')

homicides = ax.scatter(nhdf['Latitude'],nhdf['Longitude'], s=5, alpha=1,color='red', label='homicides')

tr = ax.scatter(ntdf['Latitude'],ntdf['Longitude'], s=0.3, alpha=0.4,color='yellow',label='thefts and robberies')

plt.title('Crimes Map - From 2008 to 2011')

lgnd = plt.legend(handles=[homicides, tr,other])
import matplotlib.pyplot as plt

import seaborn as sns



n3df = ndf[ndf['Year']>=2012]

nnhdf = n3df[(n3df['Primary Type']!= 'HOMICIDE') & 

            (n3df['Primary Type']!=  'THEFT') &

            (n3df['Primary Type']!= 'ROBBERY')]

nhdf = n3df[n3df['Primary Type']=='HOMICIDE']

ntdf = n3df[(n3df['Primary Type']=='THEFT')| 

           (n3df['Primary Type']=='ROBBERY')]



f,ax= plt.subplots(1,1,figsize=(12,10))

plt.xlim(41.72,42)

other = ax.scatter(nnhdf['Latitude'],nnhdf['Longitude'], s=0.4, alpha=1, label = 'other crimes')

homicides = ax.scatter(nhdf['Latitude'],nhdf['Longitude'], s=5, alpha=1,color='red', label='homicides')

tr = ax.scatter(ntdf['Latitude'],ntdf['Longitude'], s=0.3, alpha=0.4,color='yellow',label='thefts and robberies')

plt.title('Crimes Map - From 2012 to 2017')

lgnd = plt.legend(handles=[homicides, tr,other])
import seaborn as sns

homicides = ndf[ndf['Primary Type']=='HOMICIDE'].groupby('Year').count()['ID']

sns.distplot(homicides.values)