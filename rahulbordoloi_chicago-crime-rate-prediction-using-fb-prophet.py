import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import random
import seaborn as sns
from fbprophet import Prophet     #facebook prophet package
# finding out Kaggle cwd
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
chicago_df_1 = pd.read_csv('/kaggle/input/crimes-in-chicago/Chicago_Crimes_2005_to_2007.csv', error_bad_lines=False)
chicago_df_2 = pd.read_csv('/kaggle/input/crimes-in-chicago/Chicago_Crimes_2008_to_2011.csv', error_bad_lines=False)
chicago_df_3 = pd.read_csv('/kaggle/input/crimes-in-chicago/Chicago_Crimes_2012_to_2017.csv', error_bad_lines=False)
# error_bad_lines are used to ignore rows that are corrupted
#concatenating all the datasets together
chicago_df = pd.concat([chicago_df_1, chicago_df_2, chicago_df_3], ignore_index=False, axis=0)
chicago_df.shape
chicago_df.head()
chicago_df.tail(20)
#visualizing and observing the null elements in the dataset
plt.figure(figsize=(10,10))
sns.heatmap(chicago_df.isnull(), cbar = False, cmap = 'YlGnBu')   #ploting missing data #cbar, cmap = colour bar, colour map
# Dropping the following columns: ID Case Number Date Block IUCR Primary Type Description Location Description Arrest Domestic Beat District Ward Community Area FBI Code X Coordinate Y Coordinate Year Updated On Latitude Longitude Location
chicago_df.drop(['Unnamed: 0', 'Case Number', 'Case Number', 'IUCR', 'X Coordinate', 'Y Coordinate','Updated On','Year', 'FBI Code', 'Beat','Ward','Community Area', 'Location', 'District', 'Latitude' , 'Longitude'], inplace=True, axis=1)
chicago_df
#assembling a datetime by rearranging the dataframe column "Date" converting it to date-time format
chicago_df.Date = pd.to_datetime(chicago_df.Date, format='%m/%d/%Y %I:%M:%S %p')  #I-Hour %p-AM/PM
chicago_df.Date 
# setting the index to be the date-time column 
chicago_df.index = pd.DatetimeIndex(chicago_df.Date)
#counting all the no of elements within a specific column 'Primary Type'
chicago_df['Primary Type'].value_counts()
#top 15 cases
chicago_df['Primary Type'].value_counts().iloc[:15]
#indices of the top 15 cases
order_data = chicago_df['Primary Type'].value_counts().iloc[:15].index
#plotting a bar plot for the top 15 cases
plt.figure(figsize=(15,10))
sns.countplot(y='Primary Type', data=chicago_df, order = order_data)
#Locations where the crimes happened
plt.figure(figsize = (15, 10))
sns.countplot(y= 'Location Description', data = chicago_df, order = chicago_df['Location Description'].value_counts().iloc[:15].index)
#count the no of crimes occuring in a particular year
chicago_df.resample('Y').size()
#resample is a convenience method for frequency conversion and resampling of time series. 
#plotting crimmes occuring each year vs no. of crimes happening in that year
plt.plot(chicago_df.resample('Y').size())
plt.title('Crimes Count Per Year')
plt.xlabel('Years')
plt.ylabel('Number of Crimes')
chicago_df.resample('M').size()         #over the period of 'M' Months
plt.plot(chicago_df.resample('M').size())
plt.title('Crimes Count Per Month')
plt.xlabel('Months')
plt.ylabel('Number of Crimes')
chicago_df.resample('Q').size()           #over the period of 'Q' Quaters
plt.plot(chicago_df.resample('Q').size())
plt.title('Crimes Count Per Quarter')
plt.xlabel('Quarters')
plt.ylabel('Number of Crimes')
#performing quality set index
chicago_prophet = chicago_df.resample('M').size().reset_index()
chicago_prophet
chicago_prophet.columns = ['Date', 'Crime Count']
chicago_prophet
chicago_prophet_df = pd.DataFrame(chicago_prophet)
chicago_prophet_df
chicago_prophet_df.columns
#renaming the columns into 'ds' and 'y' format for facebook prophet,
#formatting in 'M' for implementation
chicago_prophet_df_final = chicago_prophet_df.rename(columns={'Date':'ds', 'Crime Count':'y'})
chicago_prophet_df_final
#instantiating prophet object
m = Prophet()
m.fit(chicago_prophet_df_final)
#forcasting into the future
future = m.make_future_dataframe(periods=720)  #periods = no. of days for prediction
forecast = m.predict(future)
forecast
#visualizing future results
figure = m.plot(forecast, xlabel='Date', ylabel='Crime Rate')
#expected trend in the future
figure3 = m.plot_components(forecast)