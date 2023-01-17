# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd  # assists with managing data frames and data series, useful data structures and methods

import numpy as np

import os

import random

import matplotlib.pyplot as plt
#read the contents of the 'petrol-prices/Petrol Prices.csv' file as a csv file and return the store data

df = pd.read_csv("../input/dwdm-petrol-prices/Petrol Prices.csv")
df
#drops the empty indexes

df = df.dropna(how='all')
#Display the first 7 records

df.head(7)
#Display the last 4 records

df.tail(4)
import datetime
#df.replace({'ug 18 2016': 'Aug 18 2016'}, regex=True)
#corrected the error in the dataset where the A was missing from Aug *ug 18 2016* in order to read same



df.at[143, 'Date'] = 'Aug 18 2016'
#change the date format of the DATE column so it is readable

df['Date']=pd.to_datetime(df.Date)
#df= df.sort_values(by='Date',ascending=True)
#print the desired index in the df

print(df.loc[[143]])
#print df

df
#Count how many records exist for each month in the data set

# result = df['Date'].groupby(df.Date.dt.strftime("%Y-%m")).agg(Total=pd.NamedAgg(column='', aggfunc='count'))

sorted_df= df.sort_values(by='Date',ascending=True)

result = sorted_df['Date'].groupby(sorted_df.Date.dt.month).agg(Total=pd.NamedAgg(column='', aggfunc='count'))

print(result)
# df['Date'].groupby(df.Date.dt.strftime("%Y-%m")).agg(Total=pd.NamedAgg(column='', aggfunc='count')).plot(kind='bar', figsize=(12,8))

result.plot(kind='bar', figsize=(12,8))

plt.title("The Number Of Records Per Month",size=8)

plt.xlabel("Month", color='orange',size=15)

plt.ylabel("Number Of Records", color='orange',size=15)
size = len(df['Date'])



months = []

days = []

years = []

timestamps = []



for i in range(size):

    date = df['Date'][i]

    

    months.append(date.strftime("%b"))

    days.append(date.day)

    years.append(date.year)

    timestamps.append(date.strftime("%d-%b-%Y %H:%M:%S"))

    

additionalDf = pd.DataFrame({

    'Month': months,

    'Day': days,

    'Year': years,

    'Timestamp': timestamps,

})



#Setting the data-types for the newly created columns 

print(additionalDf)

additionalDf[["Day", "Year"]] = additionalDf[["Day", "Year"]].apply(pd.to_numeric)

additionalDf[["Timestamp"]] = additionalDf[["Timestamp"]].apply(pd.to_datetime)
#7.Include the columns created in Task 6 with your original dataset

#pd.merge function adds the newly create columns to the existing dataset; 'left on' - where to start, 'right end' - where to end;

#how - where to start adding the new columns

merged_df = pd.merge(df, additionalDf, left_on='Date', right_on='Timestamp', how='outer')

print(merged_df)
#8.Extract the “Interesting columns” from the data set

#Creating a new dataset with the columns of interest



interestingCol = merged_df[['Date', 'Gasolene_87', 'Gasolene_90', 'Auto_Diesel', 'Kerosene']]

print(interestingCol)
#displays datatypes for each col

merged_df.dtypes
#9.Plot a line graph showing the gas prices over the particular period (hint: you may want to update the index first)

#for all the interesting columns.



#Shows the prices over the stated period. The Date col was set to the index column to achieve this

interestingCol = interestingCol.set_index('Date')

j = interestingCol[(interestingCol.index > '2019-01-01') & (interestingCol.index <= '2019-08-1')]

print(j)
j.plot(kind='line', figsize=(12,8),

    use_index = True,

    y= ['Gasolene_87','Gasolene_90','Auto_Diesel','Kerosene', ]

      )

plt.suptitle("Gas Prices Over Through Jan to Aug",size=20)

plt.xlabel("Month", color='b',size=15)

plt.ylabel("Prices", color='b',size=15)

merged_df.reset_index

merged_df.set_index('Date')



merged_df.plot(kind='line', figsize=(12,8),

    x='Date',

    y= ['Gasolene_87', 'Gasolene_90', 'Auto_Diesel', 'Kerosene', 'Propane', 'Butane', 'HFO', 'Asphalt', 'ULSD', 'Ex_Refinery'

]

      )

plt.suptitle("Gas Prices Over The Period",size=20)

plt.xlabel("Date", color='b',size=15)

plt.ylabel("Prices", color='b',size=15)

#10.Choose one interesting column and calculate the “percentage change” for every 4 time periods

#a.Display these values

#b.Plot these values using a suitable graph



interestOne = merged_df[['Gasolene_90']]

print(interestOne)

#10 continues

#fill_method : str, default ‘pad’

#How to handle NAs before computing percent changes.



interestGas = interestOne.pct_change(periods=4, fill_method='ffill')

interestGas.head(20)
interestOne.pct_change(periods=4, fill_method='ffill').plot(kind='line', figsize=(12,8),

    use_index = True, #using the index for the X axis

    y= ['Gasolene_90']

      )

plt.suptitle("Gasoline '90' Records",size=8)

plt.xlabel("Period/Frequency", color='b',size=15)

plt.ylabel("Percentage Change", color='b',size=15)
#11 Create a dataset where you select two(2) of the interesting columns and the columns created in Task 6



newSet= merged_df[['Kerosene', 'Gasolene_90', 'Month','Day','Year','Timestamp']]

newSet
#library for Kmeans imported here

from sklearn.cluster import KMeans



#specifying K. Creates KMeans object

km = KMeans(n_clusters = 3)

km
scaledNewSet = newSet
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

scaler.fit(scaledNewSet[['Kerosene']])

scaledNewSet[['Kerosene']] = scaler.transform(scaledNewSet[['Kerosene']])



scaler = MinMaxScaler()

scaler.fit(scaledNewSet[['Gasolene_90']])

scaledNewSet[['Gasolene_90']] = scaler.transform(scaledNewSet[['Gasolene_90']])

km = KMeans(n_clusters = 3)

km

y_predicted = km.fit_predict(scaledNewSet[['Kerosene', 'Gasolene_90']])

y_predicted
scaledNewSet['cluster'] = y_predicted

scaledNewSet.tail(50)
from matplotlib.pyplot import figure

scaledNewSet1 = scaledNewSet[scaledNewSet.cluster==0]

scaledNewSet2 = scaledNewSet[scaledNewSet.cluster==1]

scaledNewSet3 = scaledNewSet[scaledNewSet.cluster==2]



plt.figure(figsize=(8, 8), dpi=80)

plt.scatter(scaledNewSet1.Kerosene,scaledNewSet1.Gasolene_90,color='green')

plt.scatter(scaledNewSet2.Kerosene,scaledNewSet2.Gasolene_90,color='red')

plt.scatter(scaledNewSet3.Kerosene,scaledNewSet3.Gasolene_90,color='black')



#plotting the centroids

plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='blue',marker='*',label='centroid', s=150)





plt.xlabel('Kerosene')

plt.ylabel('Gasolene_90')

plt.legend()
#1st, 2nd and 3rd centroid of the respective clusters

km.cluster_centers_
kData = newSet[['Kerosene', 'Gasolene_90']]
k_rng = range(1,10) #setting k range

sse = []

for k in k_rng:

    km = KMeans(n_clusters=k, max_iter=300)

    km.fit(kData)

#km.inertia built in function used to calculate sse

    sse.append(km.inertia_)
#prints sse

sse
plt.xlabel('K')

plt.ylabel('Sum of squared error')

plt.plot(k_rng,sse)
km = KMeans(n_clusters=3, max_iter=300) 

kData["cluster"] = km.fit_predict( kData )

kData



kData.plot(kind='scatter',x='Kerosene',y='Gasolene_90',c=km.labels_, cmap='rainbow', figsize=(8, 8))

#plotting the centroids

plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='blue',marker='*',label='centroid', s=150)

kData['cluster'].value_counts()
kData['cluster'].value_counts().plot(kind='bar',title='Cluster for each period')
#Average per year

newSet.groupby('Year')['Kerosene','Gasolene_90'].mean()