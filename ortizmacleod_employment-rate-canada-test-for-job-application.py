#Matplotlib for data visualization

import matplotlib.pyplot as plt

#Numpy for aggregate statistics

import numpy as np

#Datetime for organizing dates

import datetime

#Pandas for data manipulation and analysis

import pandas as pd

df = pd.read_csv('../input/example_data.csv')
#First read the data: view all columns, all rows, and check if there is any missing or bad data



df.head(5)

#df.dtypes

#df.columns

#df.iloc[0:-1]

#df['month'].unique()
#Then, read Alberta and Ontario only between Jan 2007 to Nov 2019



df = df[['month','variable','sex','Alberta','Ontario']]



#df.loc[df['month'] == '2007-01']

#df.iloc[3347]



df = df.iloc[3348:-1]

df
#For the purposes of this task, we are interested in the total employment for both sexes

df = df.loc[(df['variable'] == 'Employment') & (df['sex'] == 'Both sexes')]

df
#Drop redundant columns 'variable' and 'sex' because all values equal 'Employment' and 'Both sexes' respectively



df = df.drop(columns=['variable', 'sex'])



#Reset index for simplicity



df.reset_index(drop=True, inplace=True)

df



#Dataframe now represents total employment in Alberta and Ontario, by month, from January 2007 to November 2019
#Group data by year so to calculate annual growth rate

#Column month is an object and must be changed to datetime

df.dtypes
df['month'] = pd.to_datetime(df['month'].astype(str), format='%Y%')

df
#Check to see is month column is now datetime object

df.dtypes
#Rename column month to Year, and for clarity rename Alberta to Alberta Employment and Ontario to Ontario Employment



df.columns = ['Year', 'Alberta Employment', 'Ontario Employment']
#Group data by year

grouped = df.groupby(df['Year'].map(lambda x: x.year), as_index=False)



#View months seperated by year of employment in Alberta and Ontario

for year,group in grouped:

    print (year)

    print (group)

    

#Check to see if data is still accurate

#grouped.size()

#grouped.describe()
#Next step is to create dataframe by year from Jan 2007 to Nov 2019



#Include Jan 2007 in new df so to be inclusive of annual employment growth rate since Jan 2007

start = df.iloc[0:1]

#start



#Determine last value of each year (Dec 2007, Dec 2008, etc.) to use for calculating annual growth rate

df = grouped.last()

#df



#Combine start and df dataframes so our scope is now organized by year between Jan 2007 to Nov 2019

df = start.append(df)



#Reset the index

df.reset_index(drop=True, inplace=True)

df
#Determine annual growth rate year-over-year

df[['AC%','OC%']]=df.sort_values(['Year'])[['Alberta Employment','Ontario Employment']].pct_change()*100

df
#Double check 

((2052.9-2027.6)/2027.6)*100
#Compare the percentage change data

#df['AC%'].describe()

#df['OC%'].describe()
plt.style.use('seaborn-whitegrid')

plt.figure(figsize=(15,4))

plt.plot( 'Year', 'AC%', data=df, marker='o', markerfacecolor='blue', markersize=5, color='skyblue', linewidth=4, label="Alberta")

plt.plot( 'Year', 'OC%', data=df, marker='o', markerfacecolor='olive', markersize=5, color='skyblue', linewidth=4, label="Ontario")

plt.legend()

plt.xlabel('Year')

plt.ylabel('percent')

plt.title('Annual Employment Growth Rate, Alberta and Ontario, Jan 2007 - Nov 2019')

plt.savefig('Annual_Employment_Growth_Rate_Alberta_Ontario_2007_2019.png')