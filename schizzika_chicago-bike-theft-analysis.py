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
data = pd.read_csv("/kaggle/input/Chicago_Crime_Detective.csv")
data
import dateutil

data['Date'] = data['Date'].apply(dateutil.parser.parse, dayfirst = True)
data.head()
data.isnull().any()
#data.dropna(inplace = True)
data.isnull().any()
data.dtypes
data.describe()
data.sort_values(by = 'Date', inplace = True)
data.head()
data.count()
191641/2
data.loc[95820]
data['month'] = pd.DatetimeIndex(data['Date']).month
data.head()
data.head()
data.groupby('month')['ID'].count()
data['day'] = pd.DatetimeIndex(data['Date']).day_name()
data.head()
data.groupby('day')['ID'].count()
data[data['Arrest'] == True].groupby('month').count()
data.groupby('Year')['ID'].count()
crime_by_year = pd.DataFrame({'Year':[2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012],

                              'Crime_count':[122, 494, 12977, 16823, 16403, 16069, 14271, 14280, 12039, 15484, 15622, 14003]})
crime_by_year.head()
import matplotlib.pyplot as plt

plt.plot(crime_by_year['Year'], crime_by_year['Crime_count'], linewidth = 2)

plt.xlabel('Year')

plt.ylabel('Crime count')
data[data['Arrest'] == True].groupby('Year').count()
arrest_by_year = pd.DataFrame({'Year':[2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012], 

                               'Arrest_count': [3, 23, 1415, 1691, 1523, 1301, 1212, 1013, 832, 700, 625, 547]})
arrest_by_year.head()
arrest_by_year.iloc[0:6]
arrest_by_year.iloc[6:12]
print(arrest_by_year.iloc[0:6].sum())

print(arrest_by_year.iloc[6:12].sum())
year = ['First Half', 'Second Half']

arrest = [5956, 4929]

plt.bar(year, arrest)

plt.xlabel('Duration')

plt.ylabel('Arrest count')
count_arrest_2001 = data[(data['Arrest'] == True) & (data['Year'] == 2001)]
no_arrest_2001 = data[(data['Arrest'] == False) & (data['Year'] == 2001)]
count_arrest_2001
count_arrest_2001.count()
no_arrest_2001.count()
3/121
count_arrest_2007 = data[(data['Arrest'] == True) & (data['Year'] == 2007)]
no_arrest_2007 = data[(data['Arrest'] == False) & (data['Year'] == 2007)]
count_arrest_2007.count()

no_arrest_2007.count()
1212/(1212 + 13059)
data.head()
location_wise_crime = data.groupby('LocationDescription')['ID'].count()
location_wise_crime.sort_values(ascending = False).head(10)
Top5 = location_wise_crime.sort_values(ascending = False).head(7).sum()
print(Top5)
139424 - 3270 - 1507
data.head()
data[data['LocationDescription'] == 'GAS STATION'].groupby('day').count()