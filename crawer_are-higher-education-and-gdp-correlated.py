import pandas as pd

import numpy as np

import random

import matplotlib.pyplot as plt
data = pd.read_csv('../input/Indicators.csv')
filterEducation=data['IndicatorName'].str.contains('education')

filterGDP=data['IndicatorName'].str.contains('GDP')



data[filterEducation|filterGDP]['IndicatorName'].unique().tolist()
#Choosing the indicatos "Labor force with tertiary education" and "GDP per capita" in the year 2013

filterIndicator1=data['IndicatorName']=='Labor force with tertiary education (% of total)'

filterIndicator2=data['IndicatorName']=='GDP per capita (current US$)'

filterYear=data['Year']==2013



### To make a Scatter Plot we need to assure the X and Y axis have the same size.

### So we have to chose only countries wich have both indicators 

# Filter records containing the indicator "Labor force with tertiary education"

filterEducationTemp=data[filterYear & filterIndicator1]

# Find records containing indicator "GDP per capita" within the filterEducationTemp 

filterCountryEdu=data['CountryName'].isin(filterEducationTemp['CountryName'].values)

stageGdp=data[filterCountryEdu & filterYear & filterIndicator2]

# Find records containing indicator "Labor force with tertiary education" within the stageGdp

filterCountryGdp=data['CountryName'].isin(stageGdp['CountryName'].values)

stageEducation=data[filterCountryGdp & filterYear & filterIndicator1]



# Merge stages in a final dataframe containing both indicators assuring they are indexed by CountryName

finalDataFrame=stageEducation.merge(stageGdp, on='CountryName', how='inner')
# Showing the top 10 GDP to explore our data

columnsToShow=['CountryName','IndicatorName_x','Value_x','IndicatorName_y','Value_y']

finalDataFrame[columnsToShow].sort_values(by='Value_y', ascending=False)[:10]
# Showing the top 10 Labor forces with tertiary education:

finalDataFrame[columnsToShow].sort_values(by='Value_x', ascending=False)[:10]
#Making a Scatter Plot to verify correlations 

%matplotlib inline

fig, axis = plt.subplots()



axis.yaxis.grid(True)

axis.set_title('Tertiary educated labor force vs. GDP (per capita)',fontsize=10)

axis.set_ylabel(stageGdp['IndicatorName'].iloc[0],fontsize=10)

axis.set_xlabel(stageEducation['IndicatorName'].iloc[0],fontsize=10)



Y = finalDataFrame['Value_y'].values

X = finalDataFrame['Value_x'].values



# Plot the data itself

plt.plot(X,Y,'o')



# Calc and plot the trendline

Z = np.polyfit(X, Y, 1)

P = np.poly1d(Z)

plt.plot(X,P(X),"g-")



#What is the coefficient of correlation between the two indicators ?

print ('     Coefficient of correlation:',

       np.corrcoef(finalDataFrame['Value_x'].values,finalDataFrame['Value_y'].values)[0,1])
# Excluding countries which were once socialist or communist at some point in their history

# source: https://en.wikipedia.org/wiki/List_of_former_communist_states_and_socialist_states

filterExcludedCountries=['China','Cuba','Laos','North Korea','Vietnam','Nepal','Afghanistan','Albania',

'Angola','Benin','Bulgaria','Congo-Brazzaville','Czechoslovakia','East Germany','Grenada',

'Hungary','Derg','Ethiopia','Kampuchea','Mongolia','Mozambique','North Vietnam','Poland','Romania',

'Somalia','South Yemen','Soviet Union','Yugoslavia','Armenia','Azerbaijan','Belarus',

'Estonia','Georgia','Kazakhstan','Kyrgyzstan','Latvia','Lithuania','Moldova','Russia',

'Tajikistan','Turkmenistan','Ukraine','Uzbekistan','Russian Federation','Venezuela, RB','Egypt, Arab Rep.',

'Macao SAR, China','Hong Kong SAR, China','Kyrgyz Republic']



nonSocialistsDataFrame=finalDataFrame[~finalDataFrame['CountryName'].isin(filterExcludedCountries)]



# Plotting the Scatter Plot

%matplotlib inline

fig, axis = plt.subplots()

axis.yaxis.grid(True)

axis.set_title('Tertiary educated labor force vs. GDP (per capita)',fontsize=10)

axis.set_ylabel(stageGdp['IndicatorName'].iloc[0],fontsize=10)

axis.set_xlabel(stageEducation['IndicatorName'].iloc[0],fontsize=10)

Y = nonSocialistsDataFrame['Value_y'].values

X = nonSocialistsDataFrame['Value_x'].values



# Plot the data itself

plt.plot(X,Y,'o')



# Calc and plot the trendline

Z = np.polyfit(X, Y, 1)

P = np.poly1d(Z)

plt.plot(X,P(X),"g-")



# Plotting the coefficient of correlation

print ('     Coefficient of correlation:',

       np.corrcoef(nonSocialistsDataFrame['Value_x'].values,nonSocialistsDataFrame['Value_y'].values)[0,1])