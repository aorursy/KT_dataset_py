#Python provides us libraries having built in funtion which help us to cleanise, transform and visualize the data.

#Let us inherit some of the most common libraries as below

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import math #mathematical operations/functions

%matplotlib inline 

#matplotlib is used for visualization purposes and setting it inline means 

#one need not to explicitly define the show method for visualising the graph in the notebook.

import matplotlib.pyplot as plt

import seaborn as sns #visualization purposes just like matplotlib

import datetime as dt #deal with date tiem series data



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#At this point we have imported the libraries which can later help us in a significant manner.

#As can be seen in the cell output above, we have the listed csv(comma separated file) file(s) as a data

#source on which we will will perform our analysis to extract some useful insights.
#read_csv is built in functoin provided by pandas(imported above as pd-object of pandas library) thus

#can call csv file using the below syntax.

city = pd.read_csv('../input/GlobalLandTemperaturesByCity.csv')

country = pd.read_csv('../input/GlobalLandTemperaturesByCountry.csv')

global_temp = pd.read_csv('../input/GlobalTemperatures.csv')
#The very first step to start with analysis is to know about your data.

#Let us start with extracting/checking what kind of data we have in these data files.

city.head() # head will give you the top 5 records by default- though you can pass the number by passing the parameter in the head function
city.shape #shape will give you the size of the dataframe(number of rows(observations) & number of columns(features))
country.head()
country.shape
global_temp.head()
global_temp.shape
#As we now know the size and type of data stored in all the three files, lets start with city first

#To add custom data/documentation/header like above, we use 'Markdown' cell,

#Markdown is a cell type used to display custom data/documentation/header etc.

#Click on the cell, option will appear on top-right, select the one you require and format accordingly.
unique_countries = city.Country.unique().shape[0]

unique_countries
#As I am from India, let me take the example of India and extract the dataframe containing

#climate data for India only



#This is also extracting a new dataframe from the one existing dataframe.
India = city[city.Country == 'India']

India.head()
#We have dataframe 'India' having climate data for India only from the above cell.

#pd.to_datetime convert the data to date type and dt and month ascpect gives the resulting info.

#We group the data based on the month and find the mean Avg. temp for all the cities in the India dataframe.

India_mean_temperature_monthly = India.groupby(['City', pd.to_datetime(city.dt).dt.month]).AverageTemperature.mean()



#reset_index is used to get the dataframe with mean tempratures in the Indian cities by month.

India_mean_temperature_monthly = India_mean_temperature_monthly.reset_index()



#Renaming the dataframe columns for better understanding

India_mean_temperature_monthly.columns = ['City', 'Month', 'Avg. Temp']



#Taking first 5 records, remove head to see the complete dataframe.

India_mean_temperature_monthly.head()
India_mean_temperature_monthly.shape
#As India is a large country with vast geographincal area, the different part of countries have 

#different climate/temprature. Let us take few major cities covering the different regions and plot the graph 

#for getting a clear insight of Avg temp in these cities(region).
#Create a list of cities from each region to plot their avg. monthly temp 

top_cities = ['Ahmadabad', 'Calcutta', 'Madras','New Delhi', 'Bombay', 'Srinagar']



#We select the records from above dataframe for above cities listed only.

#pivot will have three params, index, hue, column

#We have cities as index, hue means the kindof groupby based on wich data value are plotted for each index value, columns here are the monthly avg. temp value 

India_mean_temperature_monthly[India_mean_temperature_monthly.City.isin(top_cities)].pivot('City', 'Month', 'Avg. Temp').plot(kind = 'bar', figsize=(12,5), colormap = 'plasma')



#Set the title, labels and legend(loc in the legened set the legend position)

plt.title('Inidan top cities monthly average temperature')

plt.xlabel('Cities')

plt.ylabel('Average Temperature')

plt.legend(loc = 1)
India_mean_temperature_yearly = India.groupby(['City', pd.to_datetime(city.dt).dt.year]).AverageTemperature.mean().reset_index()

India_mean_temperature_yearly.columns = ['City', 'year', 'Avg. Temp']

India_mean_temperature_yearly.head()
#Get yearly avg temp for the indian top cities.

top_cities_yearly_temp = India_mean_temperature_yearly[India_mean_temperature_yearly.City.isin(top_cities)]



#Please note: we dont use . notation for accessing Avg. temp column instead use[]

#This is because if the column name has space or match with some data type/built in name, we can't use . notation

top_cities_yearly_temp.groupby('City')['Avg. Temp'].mean().plot(kind='bar')

plt.title('Indian top cities average temperature over years')

plt.xlabel('Cities')

plt.ylabel('Average Temperature')
#There is hardly a time, we can have perfect dataset for real time situation. We, therefore,

#must have to consider the missing, improper data and handle it according to our requirement.

#Let us consider the current scenario as a demonstration of this.
#The dataframe India_mean_temperature_yearly has many NaN values in the column AverageTemperature.

#We can either drop the rows with missing values or fill the missing values with mean temperatures for

#that city/month. Dropping rows will result in inconsistent rows so filling seems to be a better option.
India_mean_temperature_yearly[India_mean_temperature_yearly.City == 'Abohar']
#We are taking data for only 1 city to show the missed values in a better way.

India_mean_temperature_yearly[India_mean_temperature_yearly.City == 'Abohar']['Avg. Temp'].plot(figsize=(12,5)) #there is breaking in the line plot(because of the missing values)
#filled the missing value with mean avg temp and checking the shape

India_mean_temperature_yearly['Avg. Temp'] = India_mean_temperature_yearly['Avg. Temp'].fillna(India_mean_temperature_yearly['Avg. Temp'].mean())

India_mean_temperature_yearly.shape
#We are taking data for only 1 city to show the filling of missed values in a better way.

India_mean_temperature_yearly[India_mean_temperature_yearly.City == 'Abohar']['Avg. Temp'].plot(figsize=(12,5)) #there is no breaking in the line plot(missing values are filled)
global_temp.index = pd.to_datetime(global_temp.dt)

#resample A gives you data year wise 

global_temp.resample('A').mean().head()
global_temp.resample('A').mean()['LandAverageTemperature'].plot(figsize=(12,5))