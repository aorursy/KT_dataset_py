# Import Libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium import plugins
#Change Working directory
import os
print(os.chdir("../input"))
#Read data into dataframe df
df=pd.read_csv('2016 School Explorer.csv')
#Strips percentage string and converts it into factor
df['Percent Asian']=df['Percent Asian'].str.replace(('%'),'')
#Finding Schools with greater than 50% Asian Population
data=df[df['Percent Asian'].astype(int)>50]
#Finding Latitude and Longitude and combining them in a List
locations = data[['Latitude', 'Longitude']]
locationlist = locations.values.tolist()

#Mapping the schools on the map
map = folium.Map(location=[40.639517, -74.137071], zoom_start=11)
for point in range(0,len(locationlist) ):
    folium.Marker(locationlist[point]).add_to(map)
map
# Replacing '%' sign in each column and coercing the leftover string in integer
df['Percent Asian']=df['Percent Asian'].str.replace('%','').astype(int)
df['Percent Black']=(df['Percent Black'].str.replace(('%'),'')).astype(int)
df['Percent White']=(df['Percent White'].str.replace(('%'),'')).astype(int)
df['Percent Hispanic']=(df['Percent Hispanic'].str.replace('%','')).astype(int)

#Extracting Relevant Column defining race percentage
data=df[['Percent Asian','Percent Black','Percent White','Percent Hispanic']]
#Melting Wide form data in Long Form and storing it in data1
data1=pd.melt(data)
#Coercing Data 1 value as integer
data1.value=data1.value.astype(int)

#Plotting the Box Plot
plot=sns.boxplot(x='variable',y='value',data=data1)
plot.set_title('Distribution of Races across Classrooms')
plot.set_xlabel('Race')
plot.set_ylabel('Percentage')
#Grouping by Districts and counting the zips
plot=df.groupby(['District'])['Zip'].count().plot.bar()
plot.set_title('Districtwise School')
plot.set_xlabel('District Nos')
plot.set_ylabel('No. of Schools')
#Removing '$' & ',' from column data
df['School Income Estimate']=(df['School Income Estimate'].str.replace('$',''))
df['School Income Estimate']=(df['School Income Estimate'].str.replace(',',''))
#Converting data into float datatype
df['School Income Estimate']=df['School Income Estimate'].astype(float)

#Seperating required dataset 'data' from dataframe 'df
data=df[['Percent Asian','Percent Black','Percent White','Percent Hispanic','School Income Estimate']]
#Dropping rows without any data in School Income Estimate
data=data.dropna(subset=['School Income Estimate'])
#Finding correlation in between Races and School Income
data.corr()['School Income Estimate']
#Removing '$' & ',' from column data
df['Student Attendance Rate']=(df['Student Attendance Rate'].str.replace('%',''))
#Converting data into float datatype
df['Student Attendance Rate']=df['Student Attendance Rate'].astype(float)

#Seperating required dataset 'data' from dataframe 'df
data=df[['Percent Asian','Percent Black','Percent White','Percent Hispanic','Student Attendance Rate']]
#Dropping rows without any data in 'Student Attendance Rate'
data=data.dropna(subset=['Student Attendance Rate'])
#Finding correlation in between Races and 'Student Attendance Rate'
data.corr()['Student Attendance Rate']
#Subset of original data frame
data=df[['District','Student Achievement Rating','Zip']]
#Droping NA
data=data.dropna(subset=['District','Student Achievement Rating','Zip'])
plot=data.groupby(['District','Student Achievement Rating']).size().unstack().plot.bar(stacked=True)
plot.set_title('Districtwise Student Achievement Rating')
plot.set_ylabel('No of Schools')