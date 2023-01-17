import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline
data = pd.read_csv('../input/Traffic accidents by time of occurrence 2001-2014.csv')

print('Dataset # of rows and columns: ' + str(data.shape))

print('\nNull values per column...\n' + str(data.isnull().sum()))
data.rename(columns={'STATE/UT': 'Region', 'YEAR': 'Year', 'TYPE':'Type', '0-3 hrs. (Night)':'0_3', '3-6 hrs. (Night)':'3_6', '6-9 hrs (Day)':'6_9', '9-12 hrs (Day)':'9_12', '12-15 hrs (Day)':'12_15', '15-18 hrs (Day)':'15_18', '18-21 hrs (Night)':'18_21', '21-24 hrs (Night)':'21_24'}, inplace=True)

print('Renamed Columns: ' + str(list(data.columns)))
data.head(5)
data.sample(5)
print('Region: --> ' + str(list(data['Region'].unique())))

print('\n' + str(data['Region'].unique().size) + ' Regions in total')
print('\nYear: --> ' + str(list(data['Year'].unique())))

print('\n' + str(data['Year'].unique().size) + ' Years in total')
print('\nType: --> ' + str(list(data['Type'].unique())))

print('\n' + str(data['Type'].unique().size) + ' Types in total')
_data1 = (data[['Region','Total']].groupby('Region',as_index=False).agg('sum')).sort_values(['Total'], ascending=[False])

_data1.plot(kind='bar',x='Region',y='Total', figsize=(18,6))
_data2 = ((data[['Region','Total','Type']])[data['Type']=='Road Accidents'].groupby(['Region'], as_index=False).agg('sum')).sort_values(['Total'], ascending=[False])

_data2.plot(kind='bar',x='Region',y='Total', figsize=(18,6))
_data3 = ((data[['Region','Total','Type']])[data['Type']=='Rail-Road Accidents'].groupby(['Region'], as_index=False).agg('sum')).sort_values(['Total'], ascending=[False])

_data3.plot(kind='bar',x='Region',y='Total', figsize=(18,6))
_data4 = ((data[['Region','Total','Type']])[data['Type']=='Other Railway Accidents'].groupby(['Region'], as_index=False).agg('sum')).sort_values(['Total'], ascending=[False])

_data4.plot(kind='bar',x='Region',y='Total', figsize=(18,6))
_data5 = data[data.Total==data[data.Type=='Road Accidents']['Total'].max()]

print('Maximum Road Accidents ' + str(_data5['Total'].values[0]) + ' happened during Year ' + str(_data5['Year'].values[0]) + ' in ' + str(_data5['Region'].values[0]))
_data6 = data[data.Total==data[data.Type=='Rail-Road Accidents']['Total'].max()]

print('Maximum Rail-Road Accidents ' + str(_data6['Total'].values[0]) + ' happened during Year ' + str(_data6['Year'].values[0]) + ' in ' + str(_data6['Region'].values[0]))
_data7 = data[data.Total==data[data.Type=='Other Railway Accidents']['Total'].max()]

print('Maximum Other Railway Accidents ' + str(_data7['Total'].values[0]) + ' happened during Year ' + str(_data7['Year'].values[0]) + ' in ' + str(_data7['Region'].values[0]))
_data8 = ((data[['Region','Total','Type', 'Year']]).groupby(['Year'], as_index=False).agg('sum'))#.sort_values(['Total'], ascending=[False])

_data8.plot(kind='bar',x='Year',y='Total', figsize=(18,6))
years = list(data['Year'].unique())

totalAccidents = list(((data[['Region','Total','Type', 'Year']]).groupby(['Year'], as_index=False).agg('sum').sort_values(['Year'], ascending=[True]))['Total'])

print(totalAccidents)

#(data[['Region','Total','Type', 'Year']]).groupby(['Year'], as_index=False).agg('sum').sort_values(['Year'], ascending=[True])
# These are total registered vehicles from year 2001 to 2013 all over India

# Numbers taken from dataset: https://data.gov.in/catalog/total-number-registered-motor-vehicles-india

totalVehiclesByYear = [38556, 41581, 47519, 51922, 58799, 64743, 69129, 75336, 82402, 91598, 101865, 115419, 132550]



#The dataset doesn't have total registered vehicles for year 2014. Let's find how are numbers increasing over time....

print('Total registered vehicles number is increasing by below ratio, over years...')

i = 1

while(i < len(totalVehiclesByYear)):

    print("{:0.2f}".format(totalVehiclesByYear[i]/totalVehiclesByYear[i-1]), end=', ')

    i=i+1

print('\n\nTotal number of registered vehicles from 2001 to 2014...')



#Looks like it's close to a geometric progression. We'll consider the number of registered 

#vehicles for 2014 as 1.16 times the previous number, which seems to be fair

totalVehiclesByYear.append(int((totalVehiclesByYear[len(totalVehiclesByYear)-1])*1.16))

print(totalVehiclesByYear)
accidentsByTotalNumberOfVehicles = []

for i in range(14):

    accidentsByTotalNumberOfVehicles.append(float("{:0.2f}".format(totalAccidents[i]/totalVehiclesByYear[i])))



print('Relative accidents w.r.t., total registered vehicles from year 2001 to year 2014...')

print(accidentsByTotalNumberOfVehicles)
plt.figure(figsize=(18,6))

plt.bar(years, accidentsByTotalNumberOfVehicles)
# Helper Function

def SummarizeRegion(region):

    print('Getting data for \'' + region + '\' region...')

    data_region = (data[data['Region']==region]).drop('Total', axis=1, inplace=False)

    data_region[data_region.Type=='Road Accidents'].plot(title='Road Accidents', x='Year', kind='bar', figsize=(18,6)); 

    data_region[data_region.Type=='Rail-Road Accidents'].plot(title='Rail-Road Accidents', x='Year', kind='bar', figsize=(18,6)); 

    data_region[data_region.Type=='Other Railway Accidents'].plot(title='Other Railway Accidents', x='Year', kind='bar', figsize=(18,6)); 
SummarizeRegion('Andhra Pradesh')