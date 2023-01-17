# Load libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline 

import seaborn as sns

import folium

from folium.plugins import HeatMap

from collections import Counter

# Import data

data = pd.read_csv('../input/crime.csv', encoding='latin-1')
data.dtypes
data.head(10)

data.tail()
data.info()
data.columns
data.corr()
#correlation map

f,ax = plt.subplots(figsize=(10, 10))

sns.heatmap(data.corr(), annot=True, linewidths=.7, fmt= '.1f',ax=ax)

plt.show()
# Scatter Plot 

# x = Hour, y = Month

data.plot(kind='scatter', x='YEAR', y='MONTH',alpha = 0.1,color = 'red')

plt.xlabel('YEAR')              # label = name of label

plt.ylabel('MONTH')

plt.title('YEAR-MONTH Scatter Plot')            # title = title of plot

plt.show()
data.boxplot(column='OFFENSE_CODE',by = 'DISTRICT')

plt.show()
data['OFFENSE_CODE'].value_counts(dropna =False)

data['OFFENSE_DESCRIPTION'].value_counts(dropna =False)

data5 = data.OFFENSE_CODE

data6= data.OFFENSE_DESCRIPTION

conc_data_col = pd.concat([data5,data6],axis =1) # axis = 0 : adds dataframes in row

conc_data_col


data.boxplot(column='MONTH',by = 'YEAR')

plt.show()
# Histogram

# bins = number of bar in figure

data.HOUR.plot(kind = 'hist',bins = 100,figsize = (12,12))

plt.show()
# Histogram

# bins = number of bar in figure

data.MONTH.plot(kind = 'hist',bins = 100,figsize = (12,12))

plt.show()
#create dictionary and look its keys and values

dictionary = {'crime' : 'Vandalism','punisment' : '6-year'}

print(dictionary.keys())

print(dictionary.values())

                

series = data['DISTRICT']        # data['VANDALISM'] = series

print(type(series))

data_frame = data[['DISTRICT']]  # data[['VANDALISM']] = data frame

print(type(data_frame))
x = data['YEAR'] < 2017 # There are 265685 cases which occured before 17'

data [x]
x = data['YEAR'] > 2015 # There are 319071 cases which occured after 15'

data [x]
data[np.logical_and(data['YEAR']>2015, data['YEAR']<2017 )] # There are 99114 cases which occured between 15'-17'



data.DISTRICT[data.YEAR == 2016]
data[np.logical_and(data['YEAR'] == 2015, data['OFFENSE_DESCRIPTION'] == "HARASSMENT"  )] # There are 358 cases in 15'
data[np.logical_and(data['HOUR']>0, data['HOUR']<9 )] # There are 55189 cases in between 00:00 - 09:00
u = data.OFFENSE_CODE_GROUP

# For example lets look frequency of Offence Code Group types

data['OFFENSE_CODE_GROUP'].value_counts(dropna =False)

sta = data.OFFENSE_CODE_GROUP.value_counts().index[:10]

sns.barplot(x=sta,y = data.OFFENSE_CODE_GROUP.value_counts().values[:10])

plt.rcParams['figure.figsize']=(20,20)

plt.title('Offense Code Groups',color = 'blue',fontsize=15)

plt.show()


p = data['OFFENSE_CODE_GROUP'].value_counts(dropna =False)

plt.figure(figsize=(15,15))

sns.barplot(x=p[:8].index,y=p[:8].values)

plt.ylabel('Number of Crime')

plt.xlabel('Crime Code Group')

plt.title('Top 8 Crime Groups',color = 'red',fontsize=15)

plt.show()
for index,value in data[['DISTRICT']][0:10].iterrows(): #This data shows us top 10 crime districts.

    print(index," : ",value)
data['DISTRICT'].value_counts().plot.pie()



# Unsquish the pie.

import matplotlib.pyplot as plt

plt.gca().set_aspect('equal')


data['DISTRICT'].value_counts(dropna =True)

p = data['DISTRICT'].value_counts(dropna =True)

plt.figure(figsize=(15,15))

sns.barplot(x=p[:10].index,y=p[:10].values)

plt.ylabel('# of Cases')

plt.xlabel('District')

plt.title('Top 10 Crime District',color = 'red',fontsize=15)

plt.show()


#Crime numbers for head() districts => D14 = 20127,C11 = 42530,D4 = 41915,B3 = 35442

data2 = data.head()

crime_nm = [20127,42530,41915,41915,35442]

data2['CRIME_RATE'] = crime_nm

data2.head()
AVE_CRIME_RATE = 319073/9

data2["CRIME_CHANCE"] = ["High" if i > AVE_CRIME_RATE else "Low" for i in data2.CRIME_RATE]

data2.loc[:5,["OFFENSE_CODE_GROUP","DISTRICT","OCCURRED_ON_DATE","UCR_PART","STREET","Location","CRIME_RATE","CRIME_CHANCE"]]
plt.figure(figsize=(15,10))

sns.barplot(x=data2['DISTRICT'], y=data2['CRIME_RATE'])

plt.xticks(rotation= 45)

plt.xlabel('DISTRICTS')

plt.ylabel('CRIME_RATE')

plt.title('CRIME RATE BY SOME DISTRICTS')

plt.show()
data[np.logical_and(data['DISTRICT'] == "D14", data['OFFENSE_DESCRIPTION'] == "WARRANT ARREST" )]# There are 250 Warrant arrest cases which occured in District D14
data['REPORTING_AREA'].value_counts(dropna =False)







x = data['STREET'] == "DELHI ST" # There are 97 cases which occured İN DELHI ST

data [x]
data[np.logical_and(data['STREET'] == "DELHI ST", data['OFFENSE_DESCRIPTION'] == "INVESTIGATE PERSON" )]

data_new = data.head()

data_new
melted = pd.melt(frame=data_new,id_vars = 'STREET', value_vars= ['OFFENSE_CODE_GROUP','OFFENSE_CODE'])

melted
melted.pivot(index = 'STREET', columns = 'variable',values='value')
data1 = data[np.logical_and(data['STREET'] == "DELHI ST", data['OFFENSE_DESCRIPTION'] == "VANDALISM" )]

data2 = data[np.logical_and(data['STREET'] == "LINCOLN ST", data['OFFENSE_DESCRIPTION'] == "VANDALISM" )]



conc_data_row = pd.concat([data1,data2],axis =0,ignore_index =True) # axis = 0 : adds dataframes in row

conc_data_row
data['SHOOTING'].value_counts(dropna =False).plot.pie()



# Unsquish the pie.

import matplotlib.pyplot as plt

plt.gca().set_aspect('equal')
x = data['SHOOTING'].value_counts(dropna =False)

print(x)
data[np.logical_and(data['SHOOTING'] == "Y", data['OFFENSE_CODE_GROUP'] == "Robbery" )]

data[np.logical_and(data['SHOOTING'] == "Y", data['YEAR'] < 2020 )]

data2 = data

datetime_object = pd.to_datetime(data.OCCURRED_ON_DATE)

data2["Date_Time"] = datetime_object

# lets make date as index

data2 = data2.set_index("Date_Time")

data2 = data2.drop("OCCURRED_ON_DATE", axis=1)

data2

print(data2.loc["2018-09-02 13:00:00"])

print(data2.loc["2018-09-02 13:00:00":"2018-09-03 20:38:00"])

print(data2.loc["2017-09-02 13:00:00":"2018-09-03 20:38:00"])

data.Date_Time.sort_values()
data.index.name = "#"

data.head()
data1 = data.set_index(["YEAR","MONTH"]) 

data1