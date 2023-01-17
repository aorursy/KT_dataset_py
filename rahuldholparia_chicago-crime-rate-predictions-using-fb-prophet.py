# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from fbprophet import Prophet
df_1 = pd.read_csv("/kaggle/input/crimes-in-chicago/Chicago_Crimes_2005_to_2007.csv", error_bad_lines = False)
df_2 = pd.read_csv("/kaggle/input/crimes-in-chicago/Chicago_Crimes_2008_to_2011.csv", error_bad_lines = False)
df_3 = pd.read_csv("/kaggle/input/crimes-in-chicago/Chicago_Crimes_2012_to_2017.csv", error_bad_lines = False)
# concatenating the dataframes in to one 
df = pd.concat([df_1, df_2, df_3])
df.head()
df.shape
plt.figure(figsize = (10, 10)) 
sns.heatmap(df.isnull(), cbar = False, cmap = 'YlGnBu')
df_new = df[['Date', 'Block', 'Primary Type', 'Description', 'Location Description', 'Arrest', 'Domestic']]
df_new.head()
df_new.Date = pd.to_datetime(df_new.Date, format = '%m/%d/%Y %I:%M:%S %p')
df_new.head()
df_new['Primary Type'].value_counts().iloc[:15]
plt.figure(figsize = (15, 10))
sns.countplot(y = 'Primary Type', data = df_new, order = df_new['Primary Type'].value_counts().iloc[:15].index)
plt.figure(figsize = (15, 10))
sns.countplot(y = 'Location Description', data = df_new, order = df_new['Location Description'].value_counts().iloc[:15].index)
df_new.index = pd.DatetimeIndex(df_new.Date)
df_new.head()
df_new.resample('Y').size()
plt.plot(df_new.resample('Y').size())
plt.title('Crime count per year from 2012 to 2017')
plt.xlabel('Year')
plt.ylabel('Number of crimes')
plt.plot(df_new.resample('m').size())
plt.title('Crime count per month from 2012 to 2017')
plt.xlabel('Month')
plt.ylabel('Number of crimes')
# Reseting the index number of rows
chicago_prophet = df_new.resample('m').size().reset_index()
chicago_prophet
# changing the column names for clarity
chicago_prophet.columns = ['Date', 'Crime count']
chicago_prophet
chicago_prophet_df_final = chicago_prophet.rename(columns = {'Date': 'ds', 'Crime count': 'y'})
chicago_prophet_df_final
m = Prophet()
m.fit(chicago_prophet_df_final)
future = m.make_future_dataframe(periods = 720)
forecast = m.predict(future)
forecast
figure = m.plot(forecast, xlabel = 'Date', ylabel = 'Crime rate')
figure = m.plot_components(forecast)