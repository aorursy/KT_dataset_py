# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Prophet is the open source software released by facebook's core data science team.

# It's produced for forcasting time series data based on additive model where non-liner trends are fit with yearly, weekly,

# and daily seasonality,plus holiday effects 

# !pip install fbprophet
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from fbprophet import Prophet

import warnings

warnings.filterwarnings('ignore')
data_1 = pd.read_csv('/kaggle/input/crimes-in-chicago/Chicago_Crimes_2001_to_2004.csv',error_bad_lines=False)

data_2 = pd.read_csv('/kaggle/input/crimes-in-chicago/Chicago_Crimes_2005_to_2007.csv',error_bad_lines=False)

# data_3 = pd.read_csv('/kaggle/input/crimes-in-chicago/Chicago_Crimes_2008_to_2011.csv',error_bad_lines=False)

# data_4 = pd.read_csv('/kaggle/input/crimes-in-chicago/Chicago_Crimes_2008_to_2011.csv',error_bad_lines=False)
data_1.shape,data_2.shape
chicago_data = pd.concat([data_1,data_2])
chicago_data.shape
chicago_data.head()
#Let's check for the null values

plt.figure(figsize=(10,10))

sns.heatmap(chicago_data.isnull(),cbar=False,cmap='YlGnBu')

plt.show()
chicago_data.columns
#Drop the columns that we are not going to use from the dataset

chicago_data.drop(['Unnamed: 0','ID','Case Number','IUCR','X Coordinate', 'Y Coordinate','Updated On', 'Latitude',

       'Longitude','Location','FBI Code','Ward','Year','Community Area','District','Beat'],axis=1,inplace=True)
chicago_data.head()
chicago_data.info()
chicago_data['Date'] = pd.to_datetime(chicago_data.Date,format = '%m/%d/%Y %I:%M:%S %p')
chicago_data.info()
chicago_data.Date.head()
chicago_data.head()
chicago_data['Primary Type'].value_counts()
order_data = chicago_data['Primary Type'].value_counts().iloc[:15].index

order_data
plt.figure(figsize=(15,10))

sns.countplot(y='Primary Type',data=chicago_data,order=order_data)

plt.title('Top 15 Crims of Chicago')

plt.show()
order_for_location = chicago_data['Location Description'].value_counts().iloc[:15].index

order_for_location
plt.figure(figsize=(15,10))

sns.countplot(y='Location Description',data=chicago_data,order=order_for_location)

plt.title('Top 15 Places where happend (Crims of Chicago)')

plt.show()
chicago_data.set_index('Date',inplace=True)
chicago_data.head()
temp = chicago_data.resample('Y').size()

temp
plt.plot(temp)

plt.title('Crime Per year count')

plt.xlabel('Year')

plt.ylabel('Number of crims')

plt.show()
plt.plot(chicago_data.resample('M').size())

plt.title('Crimes per month count')

plt.xlabel('Year')

plt.ylabel('Number of Crims')

plt.show()
plt.plot(chicago_data.resample('Q').size())

plt.title('Crimes per Quaterly count')

plt.xlabel('Year')

plt.ylabel('Number of Crims')

plt.show()
chicago_prohet = chicago_data.resample('M').size().reset_index()

chicago_prohet
chicago_prohet.columns = ['Date','CrimeCount']

chicago_prohet
chicago_prohet_data = chicago_prohet.rename(columns={'Date':'ds','CrimeCount':'y'})

chicago_prohet_data
m = Prophet()

m.fit(chicago_prohet_data)
future = m.make_future_dataframe(periods=365)

forcast = m.predict(future)

forcast
figure = m.plot(forcast,xlabel='Date',ylabel='Crim Rate')
figure = m.plot_components(forcast)
future = m.make_future_dataframe(periods=720)

forcast = m.predict(future)

forcast
figure = m.plot(forcast,xlabel='Date',ylabel='Crim Rate')
figure = m.plot_components(forcast)