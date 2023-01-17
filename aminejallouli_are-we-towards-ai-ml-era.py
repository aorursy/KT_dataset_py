import pandas as pd
import numpy as np
import random
%matplotlib inline
import matplotlib.pyplot as plt
data = pd.read_csv('../input/Indicators.csv')
data.shape
countries = pd.read_csv('../input/Country.csv') 
countries.shape
# the group of countries
countries[countries['CurrencyUnit'].isnull()].head()
# creation of a series for the countries only
sr_countries_only = countries[countries['CurrencyUnit'].isnull()==False]['CountryCode']
# filter of counties
filterCountry = (data['CountryCode'].isin(sr_countries_only))
# countries data
dataCountries= data[filterCountry]
dataCountries.head()
dataCountries.loc[filterCountry,'ID']=dataCountries.loc[filterCountry,'CountryCode']+dataCountries.loc[filterCountry,'Year'].map(str)
dataCountries['ID'].isnull().any()
pd.crosstab(dataCountries['ID'].isnull(),dataCountries['CountryCode'].isnull())
dataCountries.loc[:,'CNT'] = 1
#del dataCountries['CNT']
dataCountries.loc[:5,'CNT']
# checking if the length of all the country code is 3
tmp = pd.DataFrame({'lng':dataCountries['CountryCode'].str.len()})
tmp['cnt'] = 1
tmp.groupby('lng').sum()
dataCountries[dataCountries['IndicatorCode']=='NY.GDP.MKTP.KN'].head()
GDP_lcu = dataCountries[dataCountries['IndicatorCode']=='NY.GDP.MKTP.KN'][['ID','Value']]
GDP_lcu.head()
GDP_lcu.columns=['ID', 'GDP_lcu']
GDP_lcu.shape
dataCountries.head()
# gdp per capita in usd
GDP_perCapita = dataCountries[dataCountries['IndicatorCode']=='NY.GDP.PCAP.KD'][['ID','Value']]
GDP_perCapita.columns=['ID', 'GDP_perCapita']
GDP_perCapita.shape
GDP_perCapita.head()
# Labour
labour = dataCountries[dataCountries['IndicatorCode']=='SL.TLF.TOTL.IN'][['ID','Value']]
labour.columns=['ID', 'labour']
labour.shape
labour.head()
# capital
capital = dataCountries[dataCountries['IndicatorCode']=='NE.GDI.TOTL.KN'][['ID','Value']]
capital.columns=['ID', 'capital']
capital.shape
capital.head()
# population
population = dataCountries[dataCountries['IndicatorCode']=='SP.POP.TOTL'][['ID','Value']]
population.columns=['ID', 'population']
population.shape
population.head()
consumption=dataCountries[dataCountries['IndicatorCode']=='NE.CON.TETC.CD'][['ID','Value']]
consumption.columns=['ID', 'consumption']
consumption.shape
consumption.head()
dataModel = GDP_lcu
dataModel =  pd.merge(dataModel, GDP_perCapita, on='ID', how='inner')
dataModel.shape
dataModel =  pd.merge(dataModel, labour, on='ID', how='inner')
dataModel.shape
dataModel =  pd.merge(dataModel, capital, on='ID', how='inner')
dataModel.shape
dataModel =  pd.merge(dataModel, population, on='ID', how='inner')
dataModel.shape
dataModel =  pd.merge(dataModel, consumption, on='ID', how='inner')
dataModel.shape
dataModel.head()
dataModel['Year'] = pd.to_numeric(dataModel['ID'].str.slice(3, 7)) 
dataModel.head()
dataModel['CNT']=1
dataModel.groupby('Year').sum()['CNT']
dataModel.loc[:,'GDP_usd'] = dataModel['GDP_perCapita'] * dataModel['population']
dataModel.head()
dataModel.loc[:,'Capital_usd'] = dataModel['capital']/dataModel['GDP_lcu'] * dataModel['GDP_usd']
dataModel.head()
dataModel.loc[:,'Production_usd'] = dataModel['GDP_usd'] + dataModel['consumption']
dataModel.head()
from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split 
# from sklearn.metrics import mean_squared_error
# from math import sqrt
# getting the unique years from the DataFrame dataModel
nda_years=pd.unique(dataModel['Year'])
# creating an empty dataframe
elasticities=pd.DataFrame(columns=['Year', 'ElasticityLabour','ElasticityCapital'])
elasticities.head()
features=['labour','Capital_usd']
target=['Production_usd']
dataModel.shape
dataModel.to_csv('dataModel-all-years.csv',sep=',') # some negative values were identified for the capital (6 observations).
    # these negative observations will be deleted.
dataModel = dataModel[dataModel['Capital_usd']>0]
dataModel.shape
j=0
for i in nda_years:
    # setting the year
    elasticities.loc[j,'Year'] = i
    
    # Selecting the features for the year i
    X = dataModel[dataModel['Year']==i][features]
    
    # Selecting the target for the year i
    Y = dataModel[dataModel['Year']==i][target]
    
    # Defining a linear regressor
    regressor = LinearRegression()
    regressor.fit(np.log(X),np.log(Y))
    
    # setting the elasticities
    elasticities.loc[j,'ElasticityLabour'] = regressor.coef_[0][0]
    elasticities.loc[j,'ElasticityCapital'] = regressor.coef_[0][1]
    j=j+1
elasticities
elasticities.index= elasticities['Year'].values
elasticities
plt.plot(elasticities['Year'], elasticities['ElasticityLabour'], color='blue', linewidth=1, label='Labour Elasticity')
plt.plot(elasticities['Year'], elasticities['ElasticityCapital'], color='green', linewidth=1, label='Capital Elasticity')
plt.xlabel('Year')
#plt.ylabel('Elasticities (%)')
plt.title('Labour Elasticity VS Capital Elasticity over the year 1990 to 2014')
plt.legend(loc='center left')
plt.show()
# Some data
labels = 'Labour Elasticity', 'Capital Elasticity'
fracs1990 = list(np.array(elasticities[elasticities['Year']==1990][['ElasticityLabour','ElasticityCapital']])[0])
fracs2000 = list(np.array(elasticities[elasticities['Year']==2000][['ElasticityLabour','ElasticityCapital']])[0])
fracs2014 = list(np.array(elasticities[elasticities['Year']==2014][['ElasticityLabour','ElasticityCapital']])[0])

# Make figure and axes
fig, axs = plt.subplots(3,1, figsize=(3, 15))

# A standard pie plot
axs[0].pie(fracs1990, labels=labels, autopct='%1.1f%%', shadow=True)
axs[0].set_title("Labour VS Capital in 1990")

# Shift the second slice using explode
axs[1].pie(fracs2000, labels=labels, autopct='%.0f%%', shadow=True)
axs[1].set_title("Labour VS Capital in 2000")

# Shift the second slice using explode
axs[2].pie(fracs2014, labels=labels, autopct='%.0f%%', shadow=True)
axs[2].set_title("Labour VS Capital in 2014")

plt.show()
