# Importing the important libraries for this case study

%matplotlib inline

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt 



import os

#print(os.listdir("../input"))

# Importing Data from the indicators file

data = pd.read_csv('../input/Indicators.csv')

data.head(10)
countries = data['CountryName'].unique().tolist()

codes = data['CountryCode'].unique().tolist()

indicators = data['IndicatorName'].unique().tolist()

print("Countries: %d" % (len(countries))) #Old method of printing

print("Country Codes: {}".format(len(codes)))

print("Indicators: {}".format(len(indicators)))
years = data['Year'].unique().tolist()

print(len(years))

print("Data is available from {} to {}".format(min(years), max(years)))
print(data.isnull().any().any())

print("\n")

print(data.isnull().sum())
first_indicator = 'CO2 emissions \(metric'

country = 'USA'



mask1 = data['IndicatorName'].str.contains(first_indicator) 

mask2 = data['CountryCode'].str.contains(country)



# stage is just those indicators matching the USA for country code and CO2 emissions over time.

stage_USA = data[mask1 & mask2]
#Plotting a line graph 



x_years = stage_USA['Year'].values

y_values = stage_USA['Value'].values

plt.xlabel('Years')

plt.ylabel(stage_USA['IndicatorName'].iloc[0])

plt.title('CO2 emissions per capita in USA')

#We can make a more intuitive graph by setting the axis values. You are free to comment this out

plt.axis([1959, 2011, 0, 25])  #Here, I set the y axis between 0 and 25 and x axis between 1959 to 2011



#plot function 

plt.plot(x_years, y_values)

#collecting data for India 



first_indicator = 'CO2 emissions \(metric'

country = 'India'



mask1 = data['IndicatorName'].str.contains(first_indicator) 

mask2 = data['CountryName'].str.contains(country)  #Notice how we are using Country Name here, rather than the Country Code 



stage_India = data[mask1 & mask2]
#Plotting a line graph for India



x_years = stage_India['Year'].values

y_values = stage_India['Value'].values

plt.xlabel('Years')

plt.ylabel(stage_India['IndicatorName'].iloc[0])

plt.title('CO2 emissions per capita in India')



plt.axis([1959, 2011, 0, 25])  

#If we are comparing with USA but we can also keep it to scale but since we are taking into account three countries here, I am keeping the scale constant



#plot function 

plt.plot(x_years, y_values)

#collecting data for China



first_indicator = 'CO2 emissions \(metric'

country = 'China'



mask1 = data['IndicatorName'].str.contains(first_indicator) 

mask2 = data['CountryName'].str.contains(country)  





stage_China = data[mask1 & mask2]
# Plotting a graph for China 

x_years = stage_China['Year'].values

y_values = stage_China['Value'].values



plt.xlabel('Years')

plt.ylabel(stage_China['IndicatorName'].iloc[0])

plt.title('CO2 Emissions per capita in China')



plt.axis([1959, 2011, 0, 25])

plt.plot(x_years, y_values)

# Let's look at the number of values of the China plot and see where it went wrong. 

print(len(stage_USA))

print(len(stage_India))

print(len(stage_China))



#As we saw above there are more values in the China plot. Let's see why. 

stage_China.head(15)
#Collecting Data grouped by CHN 

first_indicator = 'CO2 emissions \(metric'

country = 'CHN'



mask1 = data['IndicatorName'].str.contains(first_indicator) 

mask2 = data['CountryCode'].str.contains(country)  #changed back to CountryCode





stage_China = data[mask1 & mask2]

len(stage_China)
# Plotting a graph for China with the new grouping 



x_years = stage_China['Year'].values

y_values = stage_China['Value'].values



plt.xlabel('Years')

plt.ylabel(stage_China['IndicatorName'].iloc[0])

plt.title('CO2 Emissions per capita in China')



plt.axis([1959, 2011, 0, 25])

plt.plot(x_years, y_values)

# select GDP Per capita emissions for the United States

second_indicator = 'GDP per capita \(constant 2005'

country = 'USA'



mask1 = data['IndicatorName'].str.contains(second_indicator) 

mask2 = data['CountryCode'].str.contains(country)



# stage is just those indicators matching the USA for country code and CO2 emissions over time.

gdp_stage_USA = data[mask1 & mask2]

print(len(gdp_stage_USA))

gdp_stage_USA.head()
# select GDP Per capita emissions for the India

second_indicator = 'GDP per capita \(constant 2005'

country = 'India'



mask1 = data['IndicatorName'].str.contains(second_indicator) 

mask2 = data['CountryName'].str.contains(country)



# stage is just those indicators matching the USA for country code and CO2 emissions over time.

gdp_stage_India = data[mask1 & mask2]

print(len(gdp_stage_India))

gdp_stage_India.head()
# select GDP Per capita emissions for the India

second_indicator = 'GDP per capita \(constant 2005'

country = 'CHN'



mask1 = data['IndicatorName'].str.contains(second_indicator) 

mask2 = data['CountryCode'].str.contains(country)



# stage is just those indicators matching the USA for country code and CO2 emissions over time.

gdp_stage_China = data[mask1 & mask2]

print(len(gdp_stage_China))

gdp_stage_China.head()
# We'll need to make sure we're looking at the same time frames



print("GDP Min Year = ", gdp_stage_USA['Year'].min(), "max: ", gdp_stage_USA['Year'].max())

print("CO2 Min Year = ", stage_USA['Year'].min(), "max: ", stage_USA['Year'].max())
gdp_stage_trunc_USA = gdp_stage_USA[gdp_stage_USA['Year'] < 2012]

print(len(gdp_stage_trunc_USA))

print(len(stage_USA))



#Let's do it for the other two as well 



gdp_stage_trunc_India = gdp_stage_India[gdp_stage_India['Year'] < 2012]

gdp_stage_trunc_China = gdp_stage_China[gdp_stage_China['Year'] < 2012]

%matplotlib inline



#Plotting a subplot 

fig, axis = plt.subplots()

# Grid lines, Xticks, Xlabel, Ylabel



axis.yaxis.grid(True)

axis.set_title('CO2 Emissions vs. GDP (per capita) for USA',fontsize=10)

axis.set_xlabel(gdp_stage_trunc_USA['IndicatorName'].iloc[0],fontsize=10)   

axis.set_ylabel(stage_USA['IndicatorName'].iloc[0],fontsize=10)



X = gdp_stage_trunc_USA['Value']                   # Obtaining GDP values 

Y = stage_USA['Value']                             # Obtaining CO2 values



axis.scatter(X, Y)

plt.show()
%matplotlib inline



#Plotting a subplot

fig, axis = plt.subplots()

# Grid lines, Xticks, Xlabel, Ylabel



axis.yaxis.grid(True)

axis.set_title('CO2 Emissions vs. GDP \(per capita\) for INDIA',fontsize=10)

axis.set_xlabel(gdp_stage_trunc_India['IndicatorName'].iloc[0],fontsize=10)   

axis.set_ylabel(stage_India['IndicatorName'].iloc[0],fontsize=10)



X = gdp_stage_trunc_India['Value']                   # Obtaining GDP values 

Y = stage_India['Value']                             # Obtaining CO2 values



axis.scatter(X, Y)

plt.show()
%matplotlib inline



#Plotting a subplot

fig, axis = plt.subplots()

# Grid lines, Xticks, Xlabel, Ylabel



axis.yaxis.grid(True)

axis.set_title('CO2 Emissions vs. GDP \(per capita\)',fontsize=10)

axis.set_xlabel(gdp_stage_trunc_China['IndicatorName'].iloc[0],fontsize=10)   

axis.set_ylabel(stage_China['IndicatorName'].iloc[0],fontsize=10)



X = gdp_stage_trunc_China['Value']                   # Obtaining GDP values 

Y = stage_China['Value']                             # Obtaining CO2 values



axis.scatter(X, Y)

plt.show()
# Correlation for USA

corr_USA = np.corrcoef(gdp_stage_trunc_USA['Value'],stage_USA['Value'])

print("The correlation value for USA is: {}".format(corr_USA[1][0]))
corr_India = np.corrcoef(gdp_stage_trunc_India['Value'],stage_India['Value'])

print("The correlation value for India is: {}".format(corr_India[1][0]))
corr_China = np.corrcoef(gdp_stage_trunc_China['Value'],stage_China['Value'])

print("The correlation value for China is: {}".format(corr_China[1][0]))