import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
!ls ../input/
contries = pd.read_csv("../input/Country.csv")
contries.shape # total contries
indiaIndex = contries[contries["CountryCode"] == "IND"].index
# IND for india
contries.loc[indiaIndex]
indicators = pd.read_csv("../input/Indicators.csv")
indicators.shape # total data in the df
# create a filter for the countries
indiaMask = indicators['CountryCode'] == "IND"
usaMask = indicators['CountryCode'] == "USA"
indicators[indiaMask].head()
indicators[usaMask].head()
print(indicators[indiaMask]['Year'].unique().size)
print(indicators[usaMask]['Year'].unique().size)
# years through which the data is collected. (1960 to 2015)
# How many unique indicators do we have here. 
uniqIndicators = indicators[indiaMask]['IndicatorName'].unique()
uniqIndicatorsSize = uniqIndicators.size
print(uniqIndicatorsSize)
# all indicators with GDP in them 
# this step is just used to view all the indicators and select the most appropritate one
for i in range(uniqIndicatorsSize):
    if "GDP" in uniqIndicators[i]:
        print(uniqIndicators[i])    
# all indicators with GDP in them
# only needed to view all the indicators and to select the most appropritate one
for i in range(uniqIndicatorsSize):
    if "CO2" in uniqIndicators[i]:
        print(uniqIndicators[i])    
# In this step we create different criterias which as basically indicators that we selected from the output of the previous step
# We then use these to create masks that will be used to filter out the data from the main dataframe
# criteria
co2PerCap = "CO2 emissions \(metric tons per capita\)"
gdpPerCap = "GDP per capita \(constant 2005 US\$\)"
co2Gas = 'CO2 emissions from gaseous fuel consumption \(kt\)'
co2Liq = 'CO2 emissions from liquid fuel consumption \(kt\)'
co2Sol = 'CO2 emissions from solid fuel consumption \(kt\)'
# masks
co2PerCapMask = indicators['IndicatorName'].str.contains(co2PerCap)
gdpPerCapMask = indicators['IndicatorName'].str.contains(gdpPerCap)
co2GasMask = indicators['IndicatorName'].str.contains(co2Gas)
co2LiqMask = indicators['IndicatorName'].str.contains(co2Liq)
co2SolMask = indicators['IndicatorName'].str.contains(co2Sol)
# df with co2 per capita of india
co2Indicator = indicators[indiaMask & co2PerCapMask]
plt.figure(figsize=(12,8))
plt.plot(co2Indicator['Year'].values , co2Indicator['Value'].values, color='blue', label='CO2 per capita')
plt.xlabel("Year")
plt.ylabel(co2Indicator['IndicatorName'].iloc[0])
plt.title('CO2 Emissions in India')
plt.legend()
plt.axis([1960,2015,0,2.0])
plt.show()
gdpIndicator = indicators[indiaMask & gdpPerCapMask]
gdpIndicatorTrunc = gdpIndicator[gdpIndicator['Year']<2012]
plt.figure(figsize=(12,8))
plt.plot(gdpIndicator['Year'].values , gdpIndicator['Value'].values, 'b-', label='GDP per capita')
plt.xlabel("Year")
plt.ylabel(gdpIndicator['IndicatorName'].iloc[0])
plt.title('GDP per capita (constant 2005 US$)')
plt.legend()
plt.axis([1960,2015,0,1300])
plt.show()
plt.figure(figsize=(12,8))
plt.scatter(gdpIndicatorTrunc['Value'], co2Indicator['Value'])
plt.xlabel(gdpIndicatorTrunc['IndicatorName'].iloc[0])
plt.ylabel(co2Indicator['IndicatorName'].iloc[0])
plt.title("Correlation between GDP and CO2 emissions")
plt.axis([0,1300,0,2.0])
plt.show()
np.corrcoef(gdpIndicatorTrunc['Value'], co2Indicator['Value'])
plt.figure(figsize=(12,8))
plt.plot(indicators[co2GasMask & indiaMask]['Year'].values , indicators[co2GasMask & indiaMask]['Value'].values, 'b-', label='gaseous')
plt.plot(indicators[co2LiqMask & indiaMask]['Year'].values , indicators[co2LiqMask & indiaMask]['Value'].values, 'r-', label='liquid')
plt.plot(indicators[co2SolMask & indiaMask]['Year'].values , indicators[co2SolMask & indiaMask]['Value'].values, 'g-', label='solid')
plt.xlabel("Year")
plt.ylabel('CO2 emissions kt')
plt.title('CO2 emissions from solid, liquid and gaseous fuel in INDIA')
plt.axis([1960,2015,0,1500000])
plt.legend()
plt.show()
# indicators
co2Elec = 'CO2 emissions from electricity and heat production, total \(\% of total fuel combustion\)'
co2Manu = 'CO2 emissions from manufacturing industries and construction \(\% of total fuel combustion\)'
co2OtherEx = 'CO2 emissions from other sectors, excluding residential buildings and commercial and public services \(\% of total fuel combustion\)'
co2Public = 'CO2 emissions from residential buildings and commercial and public services \(\% of total fuel combustion\)'
co2Trans = 'CO2 emissions from transport \(\% of total fuel combustion\)'
# masks
co2ElecMask = indicators['IndicatorName'].str.contains(co2Elec)
manuMask = indicators['IndicatorName'].str.contains(co2Manu)
otherMask = indicators['IndicatorName'].str.contains(co2OtherEx)
publicMask = indicators['IndicatorName'].str.contains(co2Public)
transMask = indicators['IndicatorName'].str.contains(co2Trans)
elecMask = indicators['IndicatorName'].str.contains(co2Elec)
plt.figure(figsize=(12,8))
plt.plot(indicators[elecMask & indiaMask]['Year'].values, indicators[elecMask & indiaMask]['Value'].values, 'b-', label='Electricity and Heating')
plt.plot(indicators[manuMask & indiaMask]['Year'].values, indicators[manuMask & indiaMask]['Value'].values, 'r-', label='Manufacturing and Construction')
plt.plot(indicators[otherMask & indiaMask]['Year'].values, indicators[otherMask & indiaMask]['Value'].values, 'g-', label='Other sectors excluding resi, comm, pub')
plt.plot(indicators[publicMask & indiaMask]['Year'].values, indicators[publicMask & indiaMask]['Value'].values, 'y-', label='Residential, Commercial, Public')
plt.plot(indicators[transMask & indiaMask]['Year'].values, indicators[transMask & indiaMask]['Value'].values, color='black', label='Transport')
plt.legend(loc='best')
plt.axis([1970, 2013, 0, 60])
plt.title('CO2 emissions from economic activities in INDIA')
plt.xlabel('Year')
plt.ylabel('Percentage of total fuel consumption')
plt.show()
plt.figure(figsize=(12,8))
plt.plot(indicators[usaMask & co2PerCapMask]['Year'].values, indicators[usaMask & co2PerCapMask]['Value'].values, color='black', label='CO2 per capita')
plt.xlabel("Year")
plt.ylabel(indicators[co2PerCapMask]['IndicatorName'].iloc[0])
plt.title("CO2 emissions of USA")
plt.legend()
plt.axis([1960, 2014, 0, 25])
plt.figure(figsize=(12,8))
plt.plot(indicators[usaMask & gdpPerCapMask]['Year'].values, indicators[usaMask & gdpPerCapMask]['Value'].values, color='black', label='GDP per capita')
plt.xlabel("Year")
plt.ylabel(indicators[gdpPerCapMask]['IndicatorName'].iloc[0])
plt.title("GDP per capita of USA")
plt.legend()
plt.axis([1960, 2016, 0, 50000])
plt.figure(figsize=(12,8))
gdpUsaIndicator = indicators[usaMask & gdpPerCapMask]
gdpUsaIndicatorTrunc = gdpUsaIndicator[gdpUsaIndicator['Year']<2012]
plt.scatter(gdpUsaIndicatorTrunc['Value'].values, indicators[usaMask & co2PerCapMask]['Value'].values)
plt.show()
np.corrcoef(gdpUsaIndicatorTrunc['Value'].values, indicators[usaMask & co2PerCapMask]['Value'].values)
plt.figure(figsize=(12,8))
plt.plot(indicators[co2GasMask & usaMask]['Year'].values , indicators[co2GasMask & usaMask]['Value'].values, 'b-', label='gaseous')
plt.plot(indicators[co2LiqMask & usaMask]['Year'].values , indicators[co2LiqMask & usaMask]['Value'].values, 'r-', label='liquid')
plt.plot(indicators[co2SolMask & usaMask]['Year'].values , indicators[co2SolMask & usaMask]['Value'].values, 'g-', label='solid')
plt.xlabel("Year")
plt.ylabel('CO2 emissions kt')
plt.title('CO2 emissions from solid, liquid and gaseous fuel in USA')
plt.legend()
plt.axis([1960, 2015, 0, 2600000])
plt.show()
plt.figure(figsize=(12,8))
plt.plot(indicators[elecMask & usaMask]['Year'].values, indicators[elecMask & usaMask]['Value'].values, 'b-', label='Electricity and Heating')
plt.plot(indicators[manuMask & usaMask]['Year'].values, indicators[manuMask & usaMask]['Value'].values, 'r-', label='Manufacturing and Construction')
plt.plot(indicators[otherMask & usaMask]['Year'].values, indicators[otherMask & usaMask]['Value'].values, 'g-', label='Other sectors excluding resi, comm, pub')
plt.plot(indicators[publicMask & usaMask]['Year'].values, indicators[publicMask & usaMask]['Value'].values, 'y-', label='Residential, Commercial, Public')
plt.plot(indicators[transMask & usaMask]['Year'].values, indicators[transMask & usaMask]['Value'].values, color='black', label='Transport')
plt.legend(loc='best')
plt.axis([1960, 2013, 0, 50])
plt.title('CO2 emissions from economic activities in USA')
plt.xlabel('Year')
plt.ylabel('Percentage of total fuel consumption')
plt.show()
