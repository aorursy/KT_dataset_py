# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt  # To visualize
from sklearn import metrics
from datetime import date
from dateutil.rrule import rrule, DAILY
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from patsy import dmatrices
def linearRegressionAndEvaluation(df, Y_column_name, X_column_name, title, x_axis_name, y_axis_name):
    #create scatter plot
    df.plot.scatter(x=X_column_name, y=Y_column_name)
    X = df[X_column_name].values.reshape(-1, 1)  # values converts it into a numpy array
    Y = df[Y_column_name].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column
    
    linear_regressor = LinearRegression()  # create object for the class
    linear_regressor.fit(X, Y)  # perform linear regression
    Y_pred = linear_regressor.predict(X)  # make predictions

    # model evaluation
    rmse = metrics.mean_squared_error(Y, Y_pred)
    r2 = metrics.r2_score(Y, Y_pred)
    
    plt.title(title)
    plt.scatter(X, Y)
    plt.plot(X, Y_pred, color='red')
    plt.xlabel(x_axis_name)
    plt.ylabel(y_axis_name)
    plt.show()
    plt.savefig('books_read.png')
    
    print(f'RMSE: {rmse}')
    print(f'R2 {r2}')
    
def olsAndEvaluation(df, formula):
    #get y and x value from the dataframe and formula
    y, X = dmatrices(formula, data=df, return_type='dataframe')    
    results = sm.OLS(y, X).fit()
    print(results.summary())

    

#import the csv file with the temperatures and group by country
temperatures = pd.read_csv('/kaggle/input/climate-change-earth-surface-temperature-data/GlobalLandTemperaturesByCountry.csv')
temperatures = temperatures.groupby(['Country']).max().reset_index() #choose the maximum temperature

#read the novel corona csv and group by country.
df = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv')
df = df.groupby(['Country/Region']).sum().reset_index() #Choose the sum of the rows

#join the two data files based on country
df = df.set_index('Country/Region').join(temperatures.set_index('Country'))

df.sort_values(by=['Deaths'])

df = df.fillna(0)

#remove rows without registered deaths or temperature
df = df[(df[['AverageTemperature']] != 0).all(axis=1)]
df = df[(df[['Deaths']] != 0).all(axis=1)]

#perform the linear regression
linearRegressionAndEvaluation(df, 'Deaths', 'AverageTemperature', 'Average temperature vs amount of deaths', "Average temperature (C)", 'Amount of deaths')
olsAndEvaluation(df, 'Deaths ~ AverageTemperature')

df = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv')

#load the temperature data by country
temperatures = pd.read_csv('/kaggle/input/climate-change-earth-surface-temperature-data/GlobalLandTemperaturesByCountry.csv')
temperatures = temperatures.groupby(['Country']).max().reset_index()

#join the two tables by country
df = df.set_index('Country/Region').join(temperatures.set_index('Country'))

df['State'] = df['Province/State'].str.split(',').str[0]
df = df.groupby(['State']).max().reset_index()

#load the temperature data by state
temperatures = pd.read_csv('/kaggle/input/climate-change-earth-surface-temperature-data/GlobalLandTemperaturesByState.csv')
temperatures = temperatures.groupby(['State']).max().reset_index()

#Join the two tables and specify a suffix for the tables that overlap
df = df.set_index('State').join(temperatures.set_index('State'), lsuffix='_Country', rsuffix='_State')

def selectTemperature(df, firstChoice, secondChoice):
    df['temperature'] = np.nan
    df = df.fillna(0)
    for v in range(len(df.values)): #loop through all rows
        new_temperature = 0
        first_index = df.columns.get_loc(firstChoice)
        second_index = df.columns.get_loc(secondChoice)
        if (df.iloc[v][first_index] > 0): #if there is a value for first choose it 
            new_temperature = df.iloc[v][first_index]
        elif(df.iloc[v][second_index] >0): #else if there is a value for second choose it 
            new_temperature = df.iloc[v][second_index]
            
        df.loc[df.index.values[v],['temperature']] = new_temperature
    
    return df
#select the temperature and only keep value which are not 0
df = selectTemperature(df, 'AverageTemperature_State', 'AverageTemperature_Country')
df = df[(df[['temperature']] != 0).all(axis=1)]

#perform linear regression and print results
#linearRegressionAndEvaluation(df, 'Deaths', 'temperature', )
#olsAndEvaluation(df, 'Deaths ~ temperature')
def calculateRateOfChange(df, startDate, endDate):
    df['rate_of_change'] = np.nan #create column for rate of change
    column_in_range = False
    for v in range(len(df.values)): #loop through all the rows
        total = 0
        i = 0
        for c in range(len(df.columns)): #loop through all the columns
            if (df.columns[c] == startDate): #if the column equals the startDate set boolean on true
                column_in_range = True
            if (column_in_range): 
                if (not np.isnan(df.iloc[v,c]) and v != 0): #check if value exists
                    total = df.iloc[v,c] #create store latest value
                    i += 1 
            if (df.columns[c] == endDate):
                column_in_range = False
        if(i>0):
            rate_of_change = total**(1/float(i)) #take the nth root of total
        else:
            rate_of_change = 0
            total = 0
            i = 0
        df.loc[df.index == df.index.values[v],['rate_of_change']] = rate_of_change
    return df
#per City
df = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv')
#df = df.groupby(['location']).sum().reset_index()

#check for effect of latitude
df = calculateRateOfChange(df, '1/22/20', '4/11/20')
#linearRegressionAndEvaluation(df[(df[['rate_of_change']] != 0).all(axis=1)], 'Lat', 'rate_of_change')

df = df[(df[['4/11/20']] != 0).all(axis=1)]
df = df[(df[['rate_of_change']] != 0).all(axis=1)]

temperatures = pd.read_csv('/kaggle/input/climate-change-earth-surface-temperature-data/GlobalLandTemperaturesByCountry.csv')
temperatures = temperatures.groupby(['Country']).max().reset_index()
df = df.set_index('Country/Region').join(temperatures.set_index('Country'), lsuffix='_Country', rsuffix='_State')
df = df.fillna(0)
df = df[(df[['AverageTemperature']] != 0).all(axis=1)]
linearRegressionAndEvaluation(df, 'rate_of_change', 'AverageTemperature', "Temperature (C) vs Rate of change", "Temperature (C)", "Rate of change")
olsAndEvaluation(df, 'rate_of_change ~ AverageTemperature + Lat')
#per City
df = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths_US.csv')

#drop unescessary columns
df = df.drop(['UID', 'iso2', 'iso3', 'code3', 'FIPS'], axis='columns')

#group by the column Admin2 which contains all the cities
df = df.groupby(['Admin2']).max().reset_index()

df = calculateRateOfChange(df, '1/22/20', '4/11/20')
df = df[(df[['rate_of_change']] != 0).all(axis=1)]

#linearRegressionAndEvaluation(df, 'Lat', 'rate_of_change')
temperatures = pd.read_csv('/kaggle/input/climate-change-earth-surface-temperature-data/GlobalLandTemperaturesByState.csv')
temperatures = temperatures.loc[temperatures['Country'] == 'United States']
temperatures = temperatures.groupby(['State']).max().reset_index()
df = df.set_index('Province_State').join(temperatures.set_index('State'))
temperatures = pd.read_csv('/kaggle/input/climate-change-earth-surface-temperature-data/GlobalLandTemperaturesByCity.csv')
temperatures = temperatures.loc[temperatures['Country'] == 'United States']
temperatures = temperatures.drop(['Latitude', 'Longitude'], axis=1)
temperatures = temperatures.groupby(['City']).max().reset_index()

df = df.set_index('Admin2').join(temperatures.set_index('City'), lsuffix='_State', rsuffix='_City')
df.sort_values(by=['rate_of_change'])
df = df.fillna(0)
df = selectTemperature(df, "AverageTemperature_City", "AverageTemperature_State")
df = df[(df[['temperature']] != 0).all(axis=1)]
linearRegressionAndEvaluation(df, 'rate_of_change', 'temperature',  "Temperature (C) vs Rate of change", "Temperature (C)", "Rate of change")
olsAndEvaluation(df, 'rate_of_change ~ temperature + Lat')