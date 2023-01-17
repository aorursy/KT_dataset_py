import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
%matplotlib inline
city = pd.read_csv('../input/GlobalLandTemperaturesByCity.csv')
country = pd.read_csv('../input/GlobalLandTemperaturesByCountry.csv')
global_temp = pd.read_csv('../input/GlobalTemperatures.csv')
city.head()
city.shape
country.head()
country.shape
global_temp.head()
global_temp.shape
city.Country.value_counts()
India = city[city.Country == 'India']
India.head()
India = India.set_index('dt')
India.head()

India.index =  pd.to_datetime(India.index)
India.index.dtype
India_mean_temperature_monthly = India.groupby([India.index.month.rename('Month'),India.City])['AverageTemperature'].mean().reset_index()
India_mean_temperature_monthly.head()
top_cities = India_mean_temperature_monthly[India_mean_temperature_monthly.City.isin(['Ahmadabad', 'Calcutta', 'Madras','New Delhi', 'Bombay', 'Srinagar'])]
top_cities = top_cities.set_index('Month')
top_cities.head()
top_cities.plot(figsize =(12,5))
plt.title('Monthly Mean Temperature of Top Cities')
plt.ylabel("Average Temperature")
plt.grid(True)
India_mean_temperature_yearly = India.groupby([India.index.year.rename('Year'),India.City])['AverageTemperature'].mean().reset_index()
India_mean_temperature_yearly.head(10)
top_cities_yearly = India_mean_temperature_yearly[India_mean_temperature_yearly.City.isin(['Ahmadabad', 'Calcutta', 'Madras','New Delhi', 'Bombay', 'Srinagar'])]
top_cities_yearly = top_cities_yearly.set_index('Year')
top_cities_yearly.head()
top_cities_yearly.plot(color = 'brown',figsize =(12,5))
plt.title('Yearly Mean Temperature of Top Cities')
plt.ylabel("Average Temperature")
plt.grid(True)
top_cities_yearly = top_cities_yearly.fillna(top_cities_yearly.mean())
top_cities_yearly.tail()
top_cities_yearly.plot(color = 'green',figsize =(12,5))
plt.title('Yearly Mean Temperature of Top Cities')
plt.ylabel("Average Temperature")
plt.grid(True)
global_temp = global_temp.set_index('dt')
global_temp.index = pd.to_datetime(global_temp.index)
global_temp = global_temp.resample('A').mean()
global_temp.head()
x = global_temp.loc[:,['LandAverageTemperature']]
x.plot(figsize =(12,5))
plt.title('Global Temperature')
plt.xlabel("Year --->")
plt.ylabel("Temperature --->")
plt.grid(True)
country=country.set_index('dt')
country.index=pd.to_datetime(country.index)
country.head()
country_diff=country.groupby([country.index.year.rename('Year'),'Country']).AverageTemperature.mean().reset_index()
country_diff.head()
country_diff=country_diff.groupby(['Country']).AverageTemperature.agg(['max','min']).reset_index()
country_diff['diff']=country_diff['max']-country_diff['min']
country_diff.head()
country_list_max = country_diff.nlargest(10, columns = 'diff')
country_list_max
plt.figure(figsize=(12,5))
plt.title('Average Temperature')
plt.xlabel('Country')
plt.xticks(rotation = 90)
plt.ylabel('Highest Temperature Difference')
plt.plot(country_list_max['Country'],country_list_max['diff'],color='g')
plt.grid(True)
country_list_min = country_diff.nsmallest(10, columns = 'diff')
country_list_min
plt.figure(figsize=(12,5))
plt.title('Average Temperature')
plt.xlabel('Country')
plt.xticks(rotation = 90)
plt.ylabel('Lowest Temperature Difference')
plt.plot(country_list_min['Country'],country_list_min['diff'],color='r')
plt.grid(True)
country_filtered=country.groupby([country.index.year.rename('Year'),'Country']).AverageTemperature.mean().reset_index()
country_filtered=country_filtered[country_filtered['Year']>=1900]
country_filtered=country_filtered.groupby(['Country']).AverageTemperature.agg(['max','min']).reset_index()
country_filtered['diff']=country_filtered['max']-country_filtered['min']
country_filtered.head()
country_list_max = country_filtered.nlargest(10, columns = 'diff')
country_list_max
plt.figure(figsize=(12,5))
plt.title('Average Temperature')
plt.xlabel('Country')
plt.xticks(rotation = 90)
plt.ylabel('Highest Temperature Difference')
plt.plot(country_list_max['Country'],country_list_max['diff'],color='g')
plt.grid(True)
country_list_min = country_filtered.nsmallest(10, columns = 'diff')
country_list_min
plt.figure(figsize=(12,5))
plt.title('Average Temperature')
plt.xlabel('Country')
plt.xticks(rotation = 90)
plt.ylabel('Highest Temperature Difference')
plt.plot(country_list_min['Country'],country_list_min['diff'],color='r')
plt.grid(True)
developed = ['United States', 'United Kingdom', 'France', 'Germany', 'Japan', 'Canada', 'Switzerland', 'Norway', 'Sweden', 'South Korea', 'Australia']
developed_df=country[country.Country.isin(developed)]
developed_df=developed_df.groupby([developed_df.index.year.rename('Year'),'Country']).AverageTemperature.mean().reset_index()
developed_df.head()
developing = ['China', 'India', 'Columbia', 'Brazil', 'Mexico', 'Indonesia', 'Philippines', 'Maldives', 'Turkey', 'South Africa', 'Libya']
developing_df=country[country.Country.isin(developing)]
developing_df=developing_df.groupby([developing_df.index.year.rename('Year'),'Country']).AverageTemperature.mean().reset_index()
developing_df.head()
fig, axs = plt.subplots(ncols=2,figsize=(12,5))
sns.regplot(x='AverageTemperature',y='Year',fit_reg=True,data=developing_df, ax=axs[0])
axs[0].set(title = 'Developing Countries')
sns.regplot(x='AverageTemperature',y='Year',fit_reg=True,data=developed_df, ax=axs[1])
axs[1].set(title ='Developed Countries');
developing_df = developing_df[developing_df['Year'] > 1900]

X = developing_df['Year'].values.reshape(-1,1)
Y = developing_df['AverageTemperature']
#split dataset into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

#fit the model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#predict the test set results
y_pred_1 = regressor.predict(X_test)
y_pred_1
#coefficient and intercept for developing_df
regressor.coef_, regressor.intercept_
regressor.predict([[2025]])

developed_df = developed_df[developed_df['Year'] > 1900]

X = developed_df['Year'].values.reshape(-1,1)
Y = developed_df['AverageTemperature']
#split dataset into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

#fit the model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#predict the test set results
y_pred_2 = regressor.predict(X_test)
y_pred_2
#coefficient and intercept for developed_df
regressor.coef_, regressor.intercept_
regressor.predict([[2025]])
