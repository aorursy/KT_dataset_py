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
import matplotlib.pyplot as plt
import seaborn as sns
import fbprophet as Prophet
chicago_df_1 = pd.read_csv("../input/crimes-in-chicago/Chicago_Crimes_2001_to_2004.csv", error_bad_lines = False)
chicago_df_2 = pd.read_csv("../input/crimes-in-chicago/Chicago_Crimes_2005_to_2007.csv", error_bad_lines = False)
chicago_df_3 = pd.read_csv("../input/crimes-in-chicago/Chicago_Crimes_2008_to_2011.csv", error_bad_lines = False)
chicago_df_4 = pd.read_csv("../input/crimes-in-chicago/Chicago_Crimes_2012_to_2017.csv", error_bad_lines = False)
chicago_df_1.shape
chicago_df = pd.concat([chicago_df_1,chicago_df_2,chicago_df_3,chicago_df_4])
chicago_df.shape
chicago_df.head()
plt.figure(figsize = (10,10))
sns.heatmap(chicago_df.isnull(), cbar=False, cmap='YlGnBu')
chicago_df.drop(['Unnamed: 0','Case Number','ID','IUCR','X Coordinate', 'Y Coordinate','Updated On','Year','FBI Code','Beat','Ward','Community Area','Location','Latitude','Longitude','District'],axis = 1,inplace = True)
chicago_df.shape
chicago_df.head()
chicago_df.Date = pd.to_datetime(chicago_df.Date , format = '%m/%d/%Y %I:%M:%S %p')
chicago_df.Date
chicago_df.index = pd.DatetimeIndex(chicago_df.Date)
chicago_df['Primary Type'].value_counts()
#seeing top 15 crimes
order = chicago_df['Primary Type'].value_counts().iloc[:15].index
plt.figure(figsize = (15,10))
sns.countplot(y= 'Primary Type', data = chicago_df , order = order)
plt.figure(figsize = (15,10))
sns.countplot(y = 'Location Description', data= chicago_df , order = chicago_df['Location Description'].value_counts().iloc[:15].index)
chicago_df.resample("Y").size()
plt.plot(chicago_df.resample("Y").size())
plt.title("Crime Count Per Year");
plt.xlabel("Years");
plt.ylabel("Number of Crimes");
plt.plot(chicago_df.resample("M").size())
plt.title("Crime Count Per Month");
plt.xlabel("Month");
plt.ylabel("Number of Crimes");
plt.plot(chicago_df.resample("Q").size())
plt.title("Crime Count Per Quarter");
plt.xlabel("Quarter");
plt.ylabel("Number of Crimes");
chicago_prophet = chicago_df.resample('M').size().reset_index()
chicago_prophet
chicago_prophet.columns = ['Date','Crime Count']
chicago_prophet
#in order to apply prophet we need to have two columns Ds and Y

chicago_prophet_df = chicago_prophet.rename(columns = {'Date' : 'ds', 'Crime Count':'y'})
chicago_prophet_df
from fbprophet import Prophet
m = Prophet()
m.fit(chicago_prophet_df)
future = m.make_future_dataframe(periods=2, freq='Y')
forecast = m.predict(future)
forecast
figure = m.plot(forecast, xlabel= 'Date', ylabel= 'Crime Rate')
figure = m.plot_components(forecast)
