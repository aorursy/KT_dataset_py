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
# reading the data

virus = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv')



# preview data

virus.head()
# preview tail data

virus.tail()
# data dimesion

virus.shape

# 36598 observations, 8 features
# columns

virus.keys()
# get basic informaiton about missing values

virus.info()

# we know taht not all countries have the Province/State data.
# get summary information 

virus.describe()

# a high standard deviation means that the numbers are more spread out
# change the key/feature name for future use

virus.rename(columns={

    'Province/State': 'ProvinceState',

    'Country/Region': 'Country',

    'Last Update': 'Update'

}, inplace=True)



virus.keys()
# transfer observationDate to datetime format

virus['ObservationDate'] = pd.to_datetime(virus['ObservationDate'], format='%m/%d/%Y', errors='ignore')

virus.info() # ObservationDate is datetime64 data type
# US dataset inspection

us = virus[virus.Country == 'US']

print(us)
# The SQL is 

# select ObservationDate, ProvinceState, sum(Confirmed) as Confirmed, 

#                  sum(Deaths) as Deaths, sum(Recovered) as Recovered

#                  from data 

#                  where Country = '%s' 

#                  group by ProvinceState, ObservationDate

usByStateDate = us.groupby(['ProvinceState', 'ObservationDate']).sum()

print(usByStateDate)
# The SQL is

# select ObservationDate, sum(Confirmed) as Confirmed, sum(Deaths) as Deaths, 

#        sum(Recovered) as Recovered

#        from countryData 

#        group by ObservationDate

usByDate = usByStateDate.groupby(['ObservationDate']).sum()

print(usByDate)

# now we have the daily confirmed/deaths/recovered cases by country
# calculate the delta between observations by date

usByDate['ConfirmedNew'] = usByDate.sort_values('ObservationDate')['Confirmed'].diff().fillna(0)

usByDate['DeathsNew'] = usByDate.sort_values('ObservationDate')['Deaths'].diff().fillna(0)

usByDate['RecoveredNew'] = usByDate.sort_values('ObservationDate')['Recovered'].diff().fillna(0)

usByDate
from sklearn import preprocessing



# define common function to calculate

def groupByProvinceStateDate(country):

    '''

    Group data by ProvinceState and Date to calculate the sum of confirmed, deaths, and recovered



    Parameters

    ----------

    country : String

        The string to filter out the specified country.

    '''

    

    # because some data of the country has province and others don't.

    # hence the groupby value could fail

    # need to fill with specified string

    

    data = virus[virus.Country == country].fillna({'ProvinceState':'blank'}) # subset

    

    dataByStateDate = data.groupby(['ProvinceState', 'ObservationDate']).sum().reset_index()

    print('\n[****** The', country, 'data by province or state ******]')

    

    print(dataByStateDate)

    return dataByStateDate



def calculateMetrics(data, country):

    '''

    Group data by Date to calculate the sum of confirmed, the sum of deaths, the sum of recovered, death rate, and recovered rate. Then calculate the delta of different observations by date.

    Parameters

    ----------

    data : Pandas Dataframe

        The original dataset to be group by.

    country : String

        The string to filter out the specified country.

    '''

    

    data = data.groupby(['ObservationDate']).sum().reset_index()

        

    # we need to fix the first observation since it doesn't have the previous observation to calculate

    # so we just use itself as the delta by fillna()

    

    # ConfirmedNew is calculated as confirmed-of-today minus confirmed-of-yesterday

    data['ConfirmedNew'] = data.sort_values('ObservationDate')['Confirmed'].diff().fillna(data['Confirmed'])

    

    # DeathsNew is calculated as deaths-of-today minus deaths-of-yesterday

    data['DeathsNew'] = data.sort_values('ObservationDate')['Deaths'].diff().fillna(data['Deaths'])

    

    # RecoveredNew is calculated as recovered-of-today minus recovered-of-yesterday

    data['RecoveredNew'] = data.sort_values('ObservationDate')['Recovered'].diff().fillna(data['Recovered'])

    

    # DeathRate is calculated as deaths divided by confirmed

    data['DeathRate'] = data['Deaths'] / data['Confirmed']

    

    # RecoveredRate is calculated as recovered divided by confirmed

    data['RecoveredRate'] = data['Recovered'] / data['Confirmed']

    

    # The quantities trend in each country is different, we need to normalize it.

    column_names_to_normalize = ['Confirmed', 'Deaths', 'Recovered', 'ConfirmedNew', 'DeathsNew', 'RecoveredNew']

    column_names_normalize = ['ConfirmedN', 'DeathsN', 'RecoveredN', 'ConfirmedNewN', 'DeathsNewN', 'RecoveredNewN']

    confirmedNew = data[column_names_to_normalize].values

    confirmedNewNormal = preprocessing.MinMaxScaler().fit_transform(confirmedNew)

    temp = pd.DataFrame(confirmedNewNormal, columns=column_names_normalize, index = data.index)

    data = pd.concat([data, temp], axis=1, sort=False)

    

    # Because the country informaiton will be lost after sum(),

    # so we need to add country back to the dastaset.

    data['Country'] = country

    

    # add Day column to decide the happened day since the first confirmed cases

    data.insert(0, 'Day', range(1, len(data) + 1))

    

    # remove SNo column since we have Day column

    del data['SNo']

    

    

    print('\n[****** The', country, 'data by date ******]')

    print(data)

    return data
# the US has detailed information regarding states, so we need to group it.

usByState = groupByProvinceStateDate('US')



# calculate the delta of confirmed, deaths, recovered cases by grouping date

us = calculateMetrics(usByState, 'US')



us

# we can double-check the result by corresponding output from those two functions
us.info()
# the Brazil has detailed information regarding states since 5/21, so we need to group it.

brazilByProvince = groupByProvinceStateDate('Brazil')



# because Brazil doesn't have any confirmed cases on 2020-01-23, the first row, we need to remove it

brazilByProvince = brazilByProvince[brazilByProvince['Confirmed'] != 0]



# calculate the delta of confirmed, deaths, recovered cases by grouping date

brazil = calculateMetrics(brazilByProvince, 'Brazil')



brazil

# we can double-check the result by corresponding output from those two functions
# the Russia doesn't have detailed state or province information

# calculate the delta of confirmed, deaths, recovered cases by grouping date

russia = calculateMetrics(virus[virus.Country == 'Russia'], 'Russia')



russia

# we can double-check the result by corresponding output
# the UK has detailed information regarding provinces, so we need to group it.

ukByState = groupByProvinceStateDate('UK')



# calculate the delta of confirmed, deaths, recovered cases by grouping date

uk = calculateMetrics(ukByState, 'UK')



uk

# we can double-check the result by corresponding output from those two functions
# the Spain has detailed information regarding provinces, so we need to group it.

spainByState = groupByProvinceStateDate('Spain')



# calculate the delta of confirmed, deaths, recovered cases by grouping date

spain = calculateMetrics(spainByState, 'Spain')



spain

# we can double-check the result by corresponding output from those two functions
# the Italy has detailed information regarding provinces, so we need to group it.

italyByState = groupByProvinceStateDate('Italy')



# calculate the delta of confirmed, deaths, recovered cases by grouping date

italy = calculateMetrics(italyByState, 'Italy')



italy

# we can double-check the result by corresponding output from those two functions
# the France has detailed information regarding provinces, so we need to group it.

franceByState = groupByProvinceStateDate('France')



# calculate the delta of confirmed, deaths, recovered cases by grouping date

france = calculateMetrics(franceByState, 'France')



france

# we can double-check the result by corresponding output from those two functions
# the Germany has detailed information regarding provinces, so we need to group it.

germanyByState = groupByProvinceStateDate('Germany')



# calculate the delta of confirmed, deaths, recovered cases by grouping date

germany = calculateMetrics(germanyByState, 'Germany')



germany

# we can double-check the result by corresponding output from those two functions
# the Turkey doesn't have detailed state or province information

# calculate the delta of confirmed, deaths, recovered cases by grouping date

turkey = calculateMetrics(virus[virus.Country == 'Turkey'], 'Turkey')



turkey

# we can double-check the result by corresponding output
# the India doesn't have detailed state or province information

# calculate the delta of confirmed, deaths, recovered cases by grouping date

india = calculateMetrics(virus[virus.Country == 'India'], 'India')



india

# we can double-check the result by corresponding output
# the Iran doesn't have detailed state or province information

# calculate the delta of confirmed, deaths, recovered cases by grouping date

iran = calculateMetrics(virus[virus.Country == 'Iran'], 'Iran')



iran

# we can double-check the result by corresponding output
# the Peru doesn't have detailed state or province information

# calculate the delta of confirmed, deaths, recovered cases by grouping date

peru = calculateMetrics(virus[virus.Country == 'Peru'], 'Peru')



peru

# we can double-check the result by corresponding output
# the Canada has detailed information regarding provinces, so we need to group it.

canadaByState = groupByProvinceStateDate('Canada')



# calculate the delta of confirmed, deaths, recovered cases by grouping date

canada = calculateMetrics(canadaByState, 'Canada')



canada

# we can double-check the result by corresponding output from those two functions
# the China has detailed information regarding provinces, so we need to group it.

chinaByState = groupByProvinceStateDate('Mainland China')



# calculate the delta of confirmed, deaths, recovered cases by grouping date

china = calculateMetrics(chinaByState, 'Mainland China')



china['Country'] = 'China' # shorten the name of China

china

# we can double-check the result by corresponding output from those two functions
# the Chile has detailed information regarding provinces since 5/20, so we need to group it.

chileByState = groupByProvinceStateDate('Chile')



# calculate the delta of confirmed, deaths, recovered cases by grouping date

chile = calculateMetrics(chileByState, 'Chile')



chile

# we can double-check the result by corresponding output from those two functions
# the Saudi Arabia doesn't have detailed state or province information

# calculate the delta of confirmed, deaths, recovered cases by grouping date

arabia = calculateMetrics(virus[virus.Country == 'Saudi Arabia'], 'Saudi Arabia')



arabia

# we can double-check the result by corresponding output
# the Mexico has detailed information regarding provinces since 5/20, so we need to group it.

mexicoByState = groupByProvinceStateDate('Mexico')



# because Mexico doesn't have any confirmed cases on 2020-01-23, the first row, we need to remove it

mexicoByState = mexicoByState[mexicoByState['Confirmed'] != 0]



# calculate the delta of confirmed, deaths, recovered cases by grouping date

mexico = calculateMetrics(mexicoByState, 'Mexico')



mexico

# we can double-check the result by corresponding output from those two functions
# the Pakistan doesn't have detailed state or province information

# calculate the delta of confirmed, deaths, recovered cases by grouping date

pakistan = calculateMetrics(virus[virus.Country == 'Pakistan'], 'Pakistan')



pakistan

# we can double-check the result by corresponding output
# the Belgium doesn't have detailed state or province information

# calculate the delta of confirmed, deaths, recovered cases by grouping date

belgium = calculateMetrics(virus[virus.Country == 'Belgium'], 'Belgium')



belgium

# we can double-check the result by corresponding output
# the Qatar doesn't have detailed state or province information

# calculate the delta of confirmed, deaths, recovered cases by grouping date

qatar = calculateMetrics(virus[virus.Country == 'Qatar'], 'Qatar')



qatar

# we can double-check the result by corresponding output
# the Bangladesh doesn't have detailed state or province information

# calculate the delta of confirmed, deaths, recovered cases by grouping date

bangladesh = calculateMetrics(virus[virus.Country == 'Bangladesh'], 'Bangladesh')



bangladesh

# we can double-check the result by corresponding output
# the South Africa doesn't have detailed state or province information

# calculate the delta of confirmed, deaths, recovered cases by grouping date

africa = calculateMetrics(virus[virus.Country == 'South Africa'], 'South Africa')



africa

# we can double-check the result by corresponding output
# the Taiwan doesn't have detailed state or province information

# calculate the delta of confirmed, deaths, recovered cases by grouping date

taiwan = calculateMetrics(virus[virus.Country == 'Taiwan'], 'Taiwan')



taiwan

# we can double-check the result by corresponding output
# define custom function to add geographical informaiton

def addGeoInfo(data):

    '''

    add the continent, longitude, and latitude info by country



    Parameters

    ----------

    data : Pandas Dataframe

        The country dataset.

    '''

    country = data['Country'][0] # get the country name

    

    # we use the most cases city/province/state in the country

    # the info comes from time_series_covid19_confirmed_global.csv

    if (country in ['US', 'Brazil', 'Peru', 'Canada', 'Chile', 'Mexico']):

        

        data['Continent'] = 'America'

        

        if (country == 'US'):

            addLatLong(data, 37.0902, -95.7129)

        elif (country == 'Brazil'):

            addLatLong(data, -14.235, -51.9253)

        elif (country == 'Peru'):

            addLatLong(data, -9.19, -75.0152)

        elif (country == 'Canada'):

            addLatLong(data, 51.2538, -85.3232)

        elif (country == 'Chile'):

            addLatLong(data, -35.6751, -71.543)

        elif (country == 'Mexico'):

            addLatLong(data, 23.6345, -102.5528)

        else:

            print('Can\'t find the latitude/longitude of country:', country)

            

    elif (country in ['Russia', 'UK', 'Spain', 'Italy', 'France', 'Germany', 'Belgium']) :

        

        data['Continent'] = 'Europe'

        

        if (country == 'Russia'):

            addLatLong(data, 60, 90)

        elif (country == 'UK'):

            addLatLong(data, 49.3723, -2.3644)

        elif (country == 'Spain'):

            addLatLong(data, 40, -4)

        elif (country == 'Italy'):

            addLatLong(data, 43, 12)

        elif (country == 'France'):

            addLatLong(data, 46.2276, 2.2137)

        elif (country == 'Germany'):

            addLatLong(data, 51, 9)

        elif (country == 'Belgium'):

            addLatLong(data, 50.8333, 4)

        else:

            print('Can\'t find the latitude/longitude of country:', country)

            

    elif (country in ["Turkey", 'India', 'Iran', 'China', 'Saudi Arabia', 'Pakistan', 'Qatar', 'Bangladesh', 'Taiwan']) :

        

        data['Continent'] = 'Asia'

        

        if (country == 'Turkey'):

            addLatLong(data, 38.9637, 35.2433)

        elif (country == 'India'):

            addLatLong(data, 21, 78)

        elif (country == 'Iran'):

            addLatLong(data, 32, 53)

        elif (country == 'China'):

            addLatLong(data, 30.9756, 112.2707)

        elif (country == 'Saudi Arabia'):

            addLatLong(data, 24, 45)

        elif (country == 'Pakistan'):

            addLatLong(data, 30.3753, 69.3451)

        elif (country == 'Qatar'):

            addLatLong(data, 25.3548, 51.1839)

        elif (country == 'Bangladesh'):

            addLatLong(data, 23.685, 90.3563)

        elif (country == 'Taiwan'):

            addLatLong(data, 23.7, 121)

        else:

            print('Can\'t find the latitude/longitude of country:', country)

    elif (country in ["South Africa"]) :

        data['Continent'] = 'Africa'

        

        if (country == 'South Africa'):

            addLatLong(data, -30.5595, 22.9375)

        else:

            print('Can\'t find the latitude/longitude of country:', country)

    else:

        print('Can\'t find the country:', country)

        

def addLatLong(data, latitude, longitude):

    '''

    add the longitude and latitude info to dataframe



    Parameters

    ----------

    data : Pandas Dataframe

        The country dataset.

    latitude : Float

        A point on Earth's surface is the angle between the equatorial plane and the straight line that passes through that point and through (or close to) the center of the Earth.

    longitude : Float

        A point on Earth's surface is the angle east or west of a reference meridian to another meridian that passes through that point.

    '''

    data['Latitude'] = latitude

    data['Longitude'] = longitude

    print(data['Country'][0], ':', latitude, ',', longitude)
addGeoInfo(us)

addGeoInfo(brazil)

addGeoInfo(russia)

addGeoInfo(uk)

addGeoInfo(spain)

addGeoInfo(italy)

addGeoInfo(france)

addGeoInfo(germany)

addGeoInfo(turkey)

addGeoInfo(india)

addGeoInfo(iran)

addGeoInfo(peru)

addGeoInfo(canada)

addGeoInfo(china)

addGeoInfo(chile)

addGeoInfo(arabia)

addGeoInfo(mexico)

addGeoInfo(pakistan)

addGeoInfo(belgium)

addGeoInfo(qatar)

addGeoInfo(bangladesh)

addGeoInfo(africa)

addGeoInfo(taiwan)
print(us.info())



us.tail() # show some info
def clean(data):

    '''

    Remove the row if there is a negative number in ConfirmedNew, DeathsNew, or RecoveredNew. 

    After removing, print the logs.



    Parameters

    ----------

    data : Pandas Dataframe

        The country dataset.

    '''

    country = data.Country[0]

    

    if (len(data[data.ConfirmedNew < 0]) > 0):

        print('The value of ConfirmedNew of', country, 'is negative.')

        print(data[data.ConfirmedNew < 0])

        data.drop(data[data.ConfirmedNew < 0].index, inplace = True)

        print('ConfirmedNew cleaned up\n')

    

    if (len(data[data.DeathsNew < 0]) > 0):

        print('The value of DeathsNew of', country, 'is negative.')

        print(data[data.DeathsNew < 0])

        data.drop(data[data.DeathsNew < 0].index, inplace = True)

        print('DeathsNew cleaned up\n')

    

    if (len(data[data.RecoveredNew < 0]) > 0):

        print('The value of RecoveredNew of', country, 'is negative.')

        print(data[data.RecoveredNew < 0])

        data.drop(data[data.RecoveredNew < 0].index, inplace = True)

        print('RecoveredNew cleaned up\n')

        
# clean data correspondingly

clean(us)

clean(brazil)

clean(russia)

clean(uk)

clean(spain)

clean(italy)

clean(france)

clean(germany)

clean(turkey)

clean(india)

clean(iran)

clean(peru)

clean(canada)

clean(china)

clean(chile)

clean(arabia)

clean(mexico)

clean(pakistan)

clean(belgium)

clean(qatar)

clean(bangladesh) 

clean(africa)

clean(taiwan)
import seaborn as sns

import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

sns.set_style("darkgrid")

   

def scatterplot(data):

    '''

    draw scatter plot about related information.



    Parameters

    ----------

    data : Pandas Dataframe

        The original dataset to be group by.

    '''

    

    print('The Scatter Plot of', data.Country[0])

       

    plt.figure(figsize=(20,6))

    plt.subplot(1,2,1)

    plt.plot(data.Day, data.ConfirmedNew, '.')

    plt.xlabel('$Day$', fontsize=12)

    plt.ylabel('$ConfirmedNew$', fontsize=12)



    plt.subplot(1,2,2)

    plt.plot(data.Day, data.Confirmed, '.')

    plt.xlabel('$Day$', fontsize=12)

    plt.ylabel('$Confirmed$', fontsize=12)

    plt.show()



    plt.figure(figsize=(20,6))

    plt.subplot(1,2,1)

    plt.plot(data.RecoveredNew, data.ConfirmedNew, '.')

    plt.xlabel('$RecoveredNew$', fontsize=12)

    plt.ylabel('$ConfirmedNew$', fontsize=12)



    plt.subplot(1,2,2)

    plt.plot(data.DeathRate, data.ConfirmedNew, '.')

    plt.xlabel('$DeathsNew$', fontsize=12)

    plt.ylabel('$ConfirmedNew$', fontsize=12)

    plt.show()



    plt.figure(figsize=(20,6))

    plt.subplot(1,2,1)

    plt.plot(data.DeathRate, data.DeathsNew, '.')

    plt.xlabel('$DeathRate$', fontsize=12)

    plt.ylabel('$DeathsNew$', fontsize=12)



    plt.subplot(1,2,2)

    plt.plot(data.RecoveredRate, data.RecoveredNew, '.')

    plt.xlabel('$RecoveredRate$', fontsize=12)

    plt.ylabel('$RecoveredNew$', fontsize=12)

    plt.show()

    

    plt.figure(figsize=(20,6))

    plt.subplot(1,2,1)

    plt.plot(data.Day, data.DeathRate, '.')

    plt.xlabel('$Day$', fontsize=12)

    plt.ylabel('$DeathRate$', fontsize=12)



    plt.subplot(1,2,2)

    plt.plot(data.Day, data.RecoveredRate, '.')

    plt.xlabel('$Day$', fontsize=12)

    plt.ylabel('$RecoveredRate$', fontsize=12)

    plt.show()

    
scatterplot(us)
scatterplot(brazil)
scatterplot(russia)
scatterplot(uk)
scatterplot(spain)
scatterplot(italy)
scatterplot(france)
scatterplot(germany)
scatterplot(turkey)
scatterplot(india)
scatterplot(iran)
scatterplot(peru)
scatterplot(canada)
scatterplot(china)
scatterplot(chile)
scatterplot(arabia)
scatterplot(mexico)
scatterplot(pakistan)
scatterplot(belgium)
scatterplot(qatar)
scatterplot(bangladesh)
scatterplot(africa)
scatterplot(taiwan)
# draw distribution plot

def distribution(data):

    '''

    Draw distribution plot of ConfirmedNew, DeathsNew, and RecoveredNew



    Parameters

    ----------

    data : Pandas Dataframe

        The dataset to draw distribution

    '''

    

    country = data.Country[0]

    plt.figure(figsize=(30,6))



    plt.subplot(1,3,1)

    plt.title('Confirmed New Distribution Plot of ' + country)

    sns.distplot(data.ConfirmedNew)



    # We won't show the plot of DeathsNew of Taiwan because there will occur an error with message "You have categorical data, but your model needs something numerical. See our one hot encoding tutorial for a solution."

    # The reason is because the DeathsNew of Taiwan is only 0, 1, or 3, which will be treated as categorical data

    if (country != 'Taiwan'):

        plt.subplot(1,3,2)

        plt.title('Deaths New Distribution Plot of ' + country)

        sns.distplot(data.DeathsNew)



    plt.subplot(1,3,3)

    plt.title('Recovered New Distribution Plot of ' + country)

    sns.distplot(data.RecoveredNew)



    plt.show()
distribution(us)
distribution(brazil)
distribution(russia)
distribution(uk)
distribution(spain)
distribution(italy)
distribution(france)
distribution(germany)
distribution(turkey)
distribution(india)
distribution(iran)
distribution(peru)
distribution(canada)
distribution(china)
distribution(chile)
distribution(arabia)
distribution(mexico)
distribution(pakistan)
distribution(belgium)
distribution(qatar)
distribution(bangladesh)
distribution(africa)
distribution(taiwan)
world = pd.concat([us, brazil, russia, uk, spain, italy, france, germany, turkey, india, iran, peru, canada, china, chile, arabia, mexico, pakistan, belgium, qatar, bangladesh, africa, taiwan]) 



# show the related between box plot

plt.figure(figsize=(20,8))

plt.title('Country vs ConfirmedNew')

sns.boxplot(x=world.Country, y=world.ConfirmedNew)

plt.ylabel('Confirmed New')

plt.xlabel('Country')

plt.show()
# exclude unnecessary columns

pairdata = world[world.columns[~world.columns.isin(['ObservationDate', 'Latitude', 'Longitude', 'ConfirmedN', 'DeathsN', 'RecoveredN', 'ConfirmedNewN', 'DeathsNewN', 'RecoveredNewN'])]]



# take america as sample

america = pairdata[pairdata.Continent == 'America']



# ignore the categorical continent column

america = america[america.columns[~america.columns.isin(['Continent'])]]



# draw pairplot

sns.pairplot(america, hue='Country')
import statsmodels.api as sm



def evaluate(X, y, num):

    '''

    Perform linear regression on specified features



    Parameters

    ----------

    X : array_like

        A nobs x k array where nobs is the number of observations and k is the number of regressors. An intercept is not included by default and should be added by the user. See statsmodels.tools.add_constant.

    y : array_like

        A 1-d endogenous response variable. The dependent variable.

    num : int

        The evaluate number.

    '''

    model = sm.OLS(y, X).fit()

    print('\nModel #', num)

    print(model.summary())



def evaluateByCountry(data):

    '''    

    Evulate different models by composition of features



    Parameters

    ----------

    data : Pandas Dataframe

        The dataset to train by linear regression

    '''

    

    print('\nThe evaluation of', data.Country[0])



    

    # model #1

    X = data[['Day']]

    y = data['ConfirmedNewN']

    evaluate(X, y, 1)

    

    # model #2

    X = data[['Day', 'ConfirmedN']]

    evaluate(X, y, 2)

    

    # model #3

    X = data[['Day', 'ConfirmedN', 'DeathsN']]

    evaluate(X, y, 3)

    

    # model #4

    X = data[['Day', 'ConfirmedN', 'DeathsN', 'RecoveredN']]

    evaluate(X, y, 4)



    # model #5

    X = data[['Day', 'ConfirmedN', 'DeathsN', 'RecoveredN', 'DeathsNewN']]

    evaluate(X, y, 5)



    # model #6

    X = data[['Day', 'ConfirmedN', 'DeathsN', 'RecoveredN', 'DeathsNewN', 'RecoveredNewN']]

    evaluate(X, y, 6)



    # model #7

    X = data[['Day', 'ConfirmedN', 'DeathsN', 'RecoveredN', 'DeathsNewN', 'RecoveredNewN', 'DeathRate']]

    evaluate(X, y, 7)



    # model #8

    X = data[['Day', 'ConfirmedN', 'DeathsN', 'RecoveredN', 'DeathsNewN', 'RecoveredNewN', 'DeathRate', 'RecoveredRate']]

    evaluate(X, y, 8)

    

evaluateByCountry(us)
evaluateByCountry(italy)
evaluateByCountry(india)
from scipy import stats

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import mean_squared_error



def linearTrain(data):

    '''

    Perform linear regression on specified features



    Parameters

    ----------

    data : Pandas Dataframe

        The dataset to train by linear regression

    '''

    

    X = data[['Day', 'ConfirmedN', 'DeathsN', 'RecoveredN', 'DeathsNewN', 'RecoveredNewN']] # the features selected from model #6

    y = data['ConfirmedNewN']

    model = sm.OLS(y, X).fit()

    return model



def cor(model, data):

    '''

    Perform predict by given model and calculate the correlation



    Parameters

    ----------

    model: OLS data model

        The trained linear model

    data : Pandas Dataframe

        The dataset to predict by trained model

    '''



    # predict ConfirmedNew by corresponding model given the test data

    predictConfirmedNewN = model.predict(data[['Day', 'ConfirmedN', 'DeathsN', 'RecoveredN', 'DeathsNewN', 'RecoveredNewN']]) # the features selected from model #6

    

    print('\nThe prediction of', data.Country[0])

    

    # calculate correlation by pearson's methond

    pearson = stats.pearsonr(predictConfirmedNewN, data.ConfirmedNewN)

    correlation = abs(pearson[0]) * 100

    

    # calculate Mean Absolute Error

    mae = mean_absolute_error(data.ConfirmedNewN, predictConfirmedNewN)

    

    # calculate Mean Squared Error

    mse = mean_squared_error(data.ConfirmedNewN, predictConfirmedNewN)

    

    # calculate Mean Root Squared Error

    mqse = np.sqrt(mean_squared_error(data.ConfirmedNewN, predictConfirmedNewN))



    # print predicted info

    print('Correlation: %.3f' % correlation, ',       Mean Absolute Error: %.3f' % mae)

    print('Mean Squared Error: %.3f' % mse, ', Mean Root Squared Error:%.3f' % mqse)

    return correlation
# all country list

countryList = ['US', 'Brazil', 'Russia', 'UK', 'Spain', 'Italy', 'France', 'Germany', 'Turkey', 'India', 'Iran', 'Peru', 'Canada', 'China', 'Chile', 'Saudi Arabia', 'Mexico', 'Pakistan', 'Belgium', 'Qatar', 'Bangladesh', 'South Africa', 'Taiwan']



#define a global dataframe to for calculating the correlations

correlationScores = pd.DataFrame({'Country' : countryList,

                                  'Correlation' : [0] * 23}) # initial score is 0 for 22 countries



# show the correlation score are all 0

correlationScores
def corBar(countries, correlations, country):

    '''

    Draw barplots about the countries and correlations.



    Parameters

    ----------

    countries : list of countries

        The specified country in list.

    correlations : list of correlation score

        The calculated correlation score in list

    country : String

        the country name

    '''



    df = pd.DataFrame({'country' : countries,

                       'correlation' : correlations})



    plt.figure(figsize=(20,10))

    sns.barplot(data=df, x='country', y='correlation')

    plt.title('Correlations by Linear Model of ' + country)



    # add correlation score value to each bar

    # Now the trick is here. credit: https://stackoverflow.com/a/55866275/510320

    # plt.text() , you need to give (x,y) location , where you want to put the numbers,

    # So here index will give you x pos and data+1 will provide a little gap in y axis.

    for index,data in enumerate(correlations):

        plt.text(x=index-0.3 , y =data+1 , s='{:2.2f}'.format(data) , fontdict=dict(fontsize=16))



    plt.show()

    

   

def barBasedOn(data):

    '''

    Calculate correlations and draw plot



    Parameters

    ----------

    data : the pandas dataframe

        The specified country dataset.

    '''

    country = data.Country[0]

    countries = []

    correlations = []

    

    model = linearTrain(data)

    

    if (country != 'US'):

        corUs = cor(model, us) # calculate correlation score

        countries.append('US') # append country to show in the bar graph

        correlations.append(corUs) # append correlation score to show in the bar graph

        correlationScores.iat[0, 1] = correlationScores.iat[0, 1] + corUs  # add correlation score to corresponding country

        

    if (country != 'Brazil'):

        corBrazil = cor(model, brazil)

        countries.append('Brazil')

        correlations.append(corBrazil)

        correlationScores.iat[1, 1] = correlationScores.iat[1, 1] + corBrazil

        

    if (country != 'Russia'):

        corRussia = cor(model, russia)

        countries.append('Russia')

        correlations.append(corRussia)

        correlationScores.iat[2, 1] = correlationScores.iat[2, 1] + corRussia

        

    if (country != 'UK'):

        corUk = cor(model, uk)

        countries.append('UK')

        correlations.append(corUk)

        correlationScores.iat[3, 1] = correlationScores.iat[3, 1] + corUk

        

    if (country != 'Spain'):

        corSpain = cor(model, spain)

        countries.append('Spain')

        correlations.append(corSpain)

        correlationScores.iat[4, 1] = correlationScores.iat[4, 1] + corSpain

        

    if (country != 'Italy'):

        corItaly = cor(model, italy)

        countries.append('Italy')

        correlations.append(corItaly)

        correlationScores.iat[5, 1] = correlationScores.iat[5, 1] + corItaly

        

    if (country != 'France'):

        corFrance = cor(model, france)

        countries.append('France')

        correlations.append(corFrance)

        correlationScores.iat[6, 1] = correlationScores.iat[6, 1] + corFrance

        

    if (country != 'Germany'):

        corGermany = cor(model, germany)

        countries.append('Germany')

        correlations.append(corGermany)

        correlationScores.iat[7, 1] = correlationScores.iat[7, 1] + corGermany

        

    if (country != 'Turkey'):

        corTurkey = cor(model, turkey)

        countries.append('Turkey')

        correlations.append(corTurkey)

        correlationScores.iat[8, 1] = correlationScores.iat[8, 1] + corTurkey

        

    if (country != 'India'):

        corIndia = cor(model, india)

        countries.append('India')

        correlations.append(corIndia)

        correlationScores.iat[9, 1] = correlationScores.iat[9, 1] + corIndia

        

    if (country != 'Iran'):

        corIran = cor(model, iran)

        countries.append('Iran')

        correlations.append(corIran)

        correlationScores.iat[10, 1] = correlationScores.iat[10, 1] + corIran

        

    if (country != 'Peru'):

        corPeru = cor(model, peru)

        countries.append('Peru')

        correlations.append(corPeru)

        correlationScores.iat[11, 1] = correlationScores.iat[11, 1] + corPeru

        

    if (country != 'Canada'):

        corCanada = cor(model, canada)

        countries.append('Canada')

        correlations.append(corCanada)

        correlationScores.iat[12, 1] = correlationScores.iat[12, 1] + corCanada

        

    if (country != 'China'):

        corChina = cor(model, china)

        countries.append('China')

        correlations.append(corChina)

        correlationScores.iat[13, 1] = correlationScores.iat[13, 1] + corChina

        

    if (country != 'Chile'):

        corChile = cor(model, chile)

        countries.append('Chile')

        correlations.append(corChile)

        correlationScores.iat[14, 1] = correlationScores.iat[14, 1] + corChile

        

    if (country != 'Saudi Arabia'):

        corArabia = cor(model, arabia)

        countries.append('Saudi Arabia')

        correlations.append(corArabia)

        correlationScores.iat[15, 1] = correlationScores.iat[15, 1] + corArabia

        

    if (country != 'Mexico'):

        corMexico = cor(model, mexico)

        countries.append('Mexico')

        correlations.append(corMexico)

        correlationScores.iat[16, 1] = correlationScores.iat[16, 1] + corMexico

        

    if (country != 'Pakistan'):

        corPakistan = cor(model, pakistan)

        countries.append('Pakistan')

        correlations.append(corPakistan)

        correlationScores.iat[17, 1] = correlationScores.iat[17, 1] + corPakistan

        

    if (country != 'Belgium'):

        corBelgium = cor(model, belgium)

        countries.append('Belgium')

        correlations.append(corBelgium)

        correlationScores.iat[18, 1] = correlationScores.iat[18, 1] + corBelgium

        

    if (country != 'Qatar'):

        corQatar = cor(model, qatar)

        countries.append('Qatar')

        correlations.append(corQatar)

        correlationScores.iat[19, 1] = correlationScores.iat[19, 1] + corQatar

    

    if (country != 'Bangladesh'):

        corBangladesh = cor(model, bangladesh)

        countries.append('Bangladesh')

        correlations.append(corBangladesh)

        correlationScores.iat[20, 1] = correlationScores.iat[20, 1] + corBangladesh

        

    if (country != 'South Africa'):

        corAfrica = cor(model, africa)

        countries.append('South Africa')

        correlations.append(corAfrica)

        correlationScores.iat[21, 1] = correlationScores.iat[21, 1] + corAfrica

        

    if (country != 'Taiwan'):

        corTaiwan = cor(model, taiwan)

        countries.append('Taiwan')

        correlations.append(corTaiwan)

        correlationScores.iat[22, 1] = correlationScores.iat[22, 1] + corTaiwan

    

    # start drawing bar plots

    corBar(countries, correlations, country)

    
barBasedOn(us)
barBasedOn(brazil)
barBasedOn(russia)
barBasedOn(uk)
barBasedOn(spain)
barBasedOn(italy)
barBasedOn(france)
barBasedOn(germany)
barBasedOn(turkey)
barBasedOn(india)
barBasedOn(iran)
barBasedOn(peru)
barBasedOn(canada)
barBasedOn(china)
barBasedOn(chile)
barBasedOn(arabia)
barBasedOn(mexico)
barBasedOn(pakistan)
barBasedOn(belgium)
barBasedOn(qatar)
barBasedOn(bangladesh)
barBasedOn(africa)
barBasedOn(taiwan)
correlationScores
sortedScores = correlationScores.sort_values('Correlation')



plt.figure(figsize=(20,10))

sns.barplot(data=sortedScores, x='Country', y='Correlation')

plt.title('Overall scores of Mutual Correlation')



# add correlation score value to each bar

for index,data in enumerate(sortedScores.Correlation):

    plt.text(x=index-0.3 , y =data+15 , s=data , fontdict=dict(fontsize=16))



plt.show()

def trend(continent):

    '''

    Draw trend by continent

    

    Parameters

    ----------

    continent : the continent name

        the trend will filter by continent

    '''

    

    plt.figure(figsize=(20,10))

    sns.lineplot(data=world[world['Continent'] == continent ], x='Day', y='ConfirmedNew', hue='Country', style='Country', markers=False, dashes=False, linewidth=1.5)

    plt.show()
trend('America')
trend('Europe')
trend('Asia')
# pip install plotly==4.8.1

# reference: https://towardsdatascience.com/visualizing-worldwide-covid-19-data-using-python-plotly-maps-c0fba09a1b37

import plotly.graph_objects as go



fig = go.Figure(data=go.Choropleth(

    locationmode = "country names",

    locations = correlationScores['Country'],

    z = correlationScores['Correlation'],

    text = correlationScores['Correlation'],

    colorscale = 'matter',

    reversescale=True,

    colorbar_title = 'Correlation Score',

))



fig.update_layout(

    title_text='COVID-19 TOP 22 Mutual Correlation Scores',

    geo=dict(

        showcoastlines=True,

    ),

)



fig.show()