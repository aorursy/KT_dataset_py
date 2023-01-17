import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


##five hottest major cities in 2012
dfTempByMajorCity = pd.read_csv('../input/GlobalLandTemperaturesByMajorCity.csv',index_col='dt',parse_dates=[0])
dfTempByMajorCity[dfTempByMajorCity.index.year == 2012]['Country'=='Colombia'][['City','Country','AverageTemperature']].groupby(['City','Country']).mean().sort_values('AverageTemperature',ascending=False)
##five coldest major cities in 2012
dfTempByMajorCity[dfTempByMajorCity.index.year == 2012].groupby(['City','Country']).mean().sort_values('AverageTemperature',ascending=True).head()

##global land temperature trends since 1900
##using 5 year rolling mean to see a smoother trend 
dfGlobalTemp = pd.read_csv('../input/GlobalTemperatures.csv',index_col='dt',parse_dates=[0])
pd.rolling_mean(dfGlobalTemp[dfGlobalTemp.index.year > 1900]['LandAverageTemperature'],window=60).plot(x=dfGlobalTemp.index)
##using 5 year rolling mean to see a smoother trend
dfGlobalTemp = pd.read_csv('../input/GlobalTemperatures.csv',index_col='dt',parse_dates=[0])
pd.rolling_mean(dfGlobalTemp[dfGlobalTemp.index.year > 1900]['LandAndOceanAverageTemperature'],window=60).plot(x=dfGlobalTemp.index)
dfTempByCountry = pd.read_csv('../input/GlobalLandTemperaturesByCountry.csv',index_col='dt',parse_dates=[0])
dfTempByCountry[dfTempByCountry.index.year == 2012][['Country','AverageTemperature']].groupby(['Country']).mean().sort_values('AverageTemperature',ascending=False).head()
dfTempByCountry = pd.read_csv('../input/GlobalLandTemperaturesByCountry.csv',index_col='dt',parse_dates=[0])
dfTempByCountry[dfTempByCountry.index.year == 2012][['Country','AverageTemperature']].groupby(['Country']).mean().sort_values('AverageTemperature',ascending=True).head()