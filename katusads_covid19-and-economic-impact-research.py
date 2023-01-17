import pandas as pd
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import math
%matplotlib inline
data_input = pd.read_csv("/kaggle/input/ece657aw20asg4coronavirus/time_series_covid19_confirmed_global.csv")
data_input.head(10)
data_input.drop('Province/State', axis = 1, inplace = True)
data_input.drop('Lat', axis = 1, inplace = True)
data_input.drop('Long', axis = 1, inplace = True)
data_input['Total Cases'] = data_input.sum(axis=1)
top_six_countries = data_input.sort_values(by = ['Total Cases'], axis = 0, ascending = False)[:7]

data_france = data_input.loc[(data_input['Country/Region'] == 'France')]
data_france = data_france.sort_values(by = ['Total Cases'], axis = 0, ascending = False)[:1]


data_covid_korea = data_input.loc[(data_input['Country/Region'] == 'Korea, South')  | (data_input['Country/Region'] == 'Germany')  ]
temp = top_six_countries.append(data_covid_korea)

top_six_countries = temp.reset_index(drop = True)
top_six_countries.index +=1
top_six_countries = top_six_countries.drop(2)
top_six_countries = top_six_countries.reset_index(drop = True)
top_six_countries = top_six_countries.drop(2)
top_six_countries = top_six_countries.append(data_france)
top_six_countries = top_six_countries.reset_index(drop = True)

pd.set_option('display.max_columns', None)

subject_countries = top_six_countries
# data_france
subject_countries = subject_countries.loc[:,:'5/12/20']
subject_countries = subject_countries.reset_index(drop = True)
subject_countries
subject_countries = subject_countries.T
subject_countries
subject_countries.plot.line()
subject_countries = subject_countries.T
Economy_Data = pd.read_csv("/kaggle/input/data-economy/US_Dow Jones Industrial Average Historical Data.csv")
Economy_Data.head()
Economy_Data = Economy_Data.drop(columns = ['Open', 'High', 'Low', 'Vol.', 'Change %']).set_index('Date')[::-1].T.rename(index = {'Price': 'US'}).reset_index().rename( columns = {'index': 'Country/Region'})

Economy_Data
Italy_Data = pd.read_csv("/kaggle/input/data-economy/Italy_FTSE Italia All Share Historical Data.csv")
Italy_Data = Italy_Data.drop(columns = ['Open', 'High', 'Low', 'Vol.', 'Change %']).set_index('Date')[::-1].T.rename(index = {'Price': 'Italy'}).reset_index().rename( columns = {'index': 'Country/Region'})
Economy_Data = Economy_Data.append(Italy_Data)
Spain_Data = pd.read_csv("/kaggle/input/data-economy/Spain_IBEX 35 Historical Data.csv")
Spain_Data = Spain_Data.drop(columns = ['Open', 'High', 'Low', 'Vol.', 'Change %']).set_index('Date')[::-1].T.rename(index = {'Price': 'Spain'}).reset_index().rename( columns = {'index': 'Country/Region'})
Economy_Data = Economy_Data.append(Spain_Data)
Germany_Data = pd.read_csv("/kaggle/input/data-economy/Germany_DAX Historical Data.csv")
Germany_Data = Germany_Data.drop(columns = ['Open', 'High', 'Low', 'Vol.', 'Change %']).set_index('Date')[::-1].T.rename(index = {'Price': 'Germany'}).reset_index().rename( columns = {'index': 'Country/Region'})
Economy_Data = Economy_Data.append(Germany_Data)
France_Data = pd.read_csv("/kaggle/input/data-economy/France_CAC 40 Historical Data.csv")
France_Data = France_Data.drop(columns = ['Open', 'High', 'Low', 'Vol.', 'Change %']).set_index('Date')[::-1].T.rename(index = {'Price': 'France'}).reset_index().rename( columns = {'index': 'Country/Region'})
Economy_Data = Economy_Data.append(France_Data)
UK_Data = pd.read_csv("/kaggle/input/data-economy/UK_FTSE 100 Historical Data.csv")
UK_Data = UK_Data.drop(columns = ['Open', 'High', 'Low', 'Vol.', 'Change %']).set_index('Date')[::-1].T.rename(index = {'Price': 'UK'}).reset_index().rename( columns = {'index': 'Country/Region'})
Economy_Data = Economy_Data.append(UK_Data)
India_Data = pd.read_csv("/kaggle/input/data-economy/India_BSE Sensex 30 Historical Data.csv")
India_Data = India_Data.drop(columns = ['Open', 'High', 'Low', 'Vol.', 'Change %']).set_index('Date')[::-1].T.rename(index = {'Price': 'India'}).reset_index().rename( columns = {'index': 'Country/Region'})
Economy_Data = Economy_Data.append(India_Data)
Korea_Data = pd.read_csv("/kaggle/input/data-economy/Korea_KOSPI Historical Data.csv")
Korea_Data = Korea_Data.drop(columns = ['Open', 'High', 'Low', 'Vol.', 'Change %']).set_index('Date')[::-1].T.rename(index = {'Price': 'Korea'}).reset_index().rename( columns = {'index': 'Country/Region'})
Economy_Data = Economy_Data.append(Korea_Data)
Economy_Data.reset_index(drop=True)
Economy_Data.set_index('Country/Region')
Economy_Data.drop(['Feb 01, 2020', 'Apr 10, 2020', 'Feb 17, 2020'], axis=1, inplace = True)

Economy_Data.set_index('Country/Region', inplace = True)
Economy_Data = Economy_Data.replace(',','',regex=True).astype('float')
Economy_Data.interpolate(axis = 1, inplace = True)
Economy_Data
Economy_Data.T.plot.line()
subject_countries
subject_countries.drop(['1/25/20', '1/26/20', '2/1/20', '2/2/20', '2/8/20', '2/9/20', '2/15/20', '2/16/20','2/17/20','2/22/20','2/23/20','2/29/20', '3/1/20','3/7/20','3/8/20','3/14/20','3/15/20','3/21/20','3/22/20','3/28/20', '3/29/20', '4/4/20','4/5/20','4/10/20','4/11/20','4/12/20','4/18/20','4/19/20','4/25/20','4/26/20','5/2/20','5/3/20','5/9/20','5/10/20'], axis = 1, inplace = True)
# subject_countries.drop(['1/25/20', '1/26/20', '2/1/20', '2/2/20', '2/8/20', '2/9/20', '2/15/20', '2/16/20','2/17/20','2/22/20','2/23/20','2/29/20', '3/1/20','3/7/20'], axis = 1, inplace = True)
subject_countries.set_index('Country/Region', inplace = True)
subject_countries
def correlation(x, y):
    n = len(x)
    vals = range(n)

    x_sum = 0.0
    y_sum = 0.0
    x_sum_pow = 0.0
    y_sum_pow = 0.0
    mul_xy_sum = 0.0
    
    for i in vals:
        mul_xy_sum = mul_xy_sum + float(x[i]) * float(y[i])
        x_sum = x_sum + float(x[i])
        y_sum = y_sum + float(y[i])
        x_sum_pow = x_sum_pow + pow(float(x[i]), 2)
        y_sum_pow = y_sum_pow + pow(float(y[i]), 2)
        
    
    try:
        r = ((n * mul_xy_sum) - (x_sum * y_sum)) / math.sqrt( ((n*x_sum_pow) - pow(x_sum, 2)) * ((n*y_sum_pow) - pow(y_sum, 2)) )
    except:
        r = 0.0
    
    return r
Coefficient_US = correlation(list(subject_countries.loc['US']), list(Economy_Data.loc['US']))
Coefficient_Italy = correlation(list(subject_countries.loc['Italy']), list(Economy_Data.loc['Italy']))
Coefficient_Spain = correlation(list(subject_countries.loc['Spain']), list(Economy_Data.loc['Spain']))
Coefficient_UK = correlation(list(subject_countries.loc['United Kingdom']), list(Economy_Data.loc['UK']))
Coefficient_Germany = correlation(list(subject_countries.loc['Germany']), list(Economy_Data.loc['Germany']))
Coefficient_France = correlation(list(subject_countries.loc['France']), list(Economy_Data.loc['France']))
Coefficient_India = correlation(list(subject_countries.loc['India']), list(Economy_Data.loc['India']))
Coefficient_Korea = correlation(list(subject_countries.loc['Korea, South']), list(Economy_Data.loc['Korea']))
print("US: ", Coefficient_US)
print("Italy: ", Coefficient_Italy)
print("Spain: ", Coefficient_Spain)
print("UK: ", Coefficient_UK)
print("Germany: ", Coefficient_Germany)
print("France: ", Coefficient_France)
print("India: ", Coefficient_India)
print("Korea: ", Coefficient_Korea)
cor = pd.DataFrame({'Country':['US','Italy', 'Spain', 'UK', 'Germany', 'France', 'India', 'Korea'], 'val': [-0.30,-0.66,-0.67,-0.44,-0.46,-0.53,-0.35,-0.79 ]})
cor.plot.bar(x='Country', y = 'val', rot =0)

