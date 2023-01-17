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
import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt 

%matplotlib inline 

import seaborn as sns 

import folium 

from folium.plugins import HeatMap

from fbprophet import Prophet
data=pd.read_csv('../input/crimes-in-boston/crime.csv',encoding='latin-1')
data.head()
data.shape
data1 = data.rename(columns={'OFFENSE_CODE':'Code','OFFENSE_CODE_GROUP':'Group','OFFENSE_DESCRIPTION':'Description','OCCURRED_ON_DATE':'Date'})

data1.head()
plt.figure(figsize=(10,10))

sns.heatmap(data1.isnull(),cbar=False,cmap='YlGnBu')

plt.ioff()
data1.drop(['INCIDENT_NUMBER','Code','SHOOTING','UCR_PART','Lat','Long','Location'],inplace=True,axis=1)

data1.head()
data1['Date']=pd.to_datetime(data1['Date'])

data1.head()
data1.Date
data1.index=pd.DatetimeIndex(data1.Date)

data1.head()
data1['Group'].value_counts()
data1['Group'].value_counts().iloc[:15]
order_data=data1['Group'].value_counts().iloc[:15].index

plt.figure(figsize=(15,10))

sns.countplot(y='Group',data=data1,order=order_data)

plt.ioff()
data1.resample('Y').size()
plt.plot(data1.resample('Y').size())

plt.title('Crime Count Per Year')

plt.xlabel('Years')

plt.ylabel('Number of Crimes')

plt.ioff()
plt.plot(data1.resample('M').size())

plt.title('Crime Count Per Month')

plt.xlabel('Months')

plt.ylabel('Number of Crimes')

plt.ioff()
plt.plot(data1.resample('Q').size())

plt.title('Crime Count Per Quarter')

plt.xlabel('Quaterly')

plt.ylabel('Number of Crimes')

plt.ioff()
Boston_prophet=data1.resample('M').size().reset_index()
Boston_prophet.head()
Boston_prophet.columns=['Date','Crime_Count']
Boston_prophet.head()
Boston_prophet_final=Boston_prophet.rename(columns={'Date':'ds','Crime_Count':'y'})
Boston_prophet_final.head()
m=Prophet()

m.fit(Boston_prophet_final)
future=m.make_future_dataframe(periods=365)

forecast=m.predict(future)
forecast
#figure=m.plot(forecast,xlabel='Data',ylabel='Crime Rate')
#figure=m.plot_components(forecast)