#import Required modles

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

pd.set_option('display.expand_frame_repr', True)

import seaborn as sns
#load the data

df=pd.read_csv('../input/NYSERDA_DER_Metric_Data.csv')

df.head()
#lets have a look at some basic attributes of the dataset we have

#shape of the dataframe

print('The shape of the DataFrame is ', df.shape)

print('-'*100)

#info of the dataframe

print(df.info())

print('-'*100)

#columns of the dataframe

print('The columns of df are\n',df.columns.tolist())

print('-'*100)

#finding any missing values

print('No of Missing Values per column\n', df.isnull().sum())



print('-'*100)

#looking for unique values per column

print('Unique values per columns\n',df.nunique())
#lets us now take care of the other missing values

#lets visualize the missing data distribution



import missingno as msn



msn.matrix(df)

#dropping Missing Values from the selected columns

df1=df.drop(columns=['Address Line 2','Alternate Name(s)','Facility Website (external)','Floor Area (ft²)', 'No. of Occupancy Units','Gas Utility',  'Located in Flood Plain'])

df1.head(5)

print(df1.shape)
#Filling missing values by its preceding term

df1=df1.iloc[1:]

df1=df1.fillna(method='ffill')

df1.head()
# Checking the updated dataset 

msn.matrix(df1)
#just for futrue reference i have downloaded the dataset as an excel to my local

df1.to_excel('submissionnineleaps1396.xlsx')
# If we check in the dataset, we can see their are few NaN values in the subcategory column. Since subcategory column is a categorical column we will use mode to fill in the Nanvalues

df1['Subcategory'].head()

df1['Subcategory'].transform(lambda x: x.fillna(x.mode()[0])).head()
#Lets look at the data set again

df1.head()
#analsing the city with highest electric power generation

f,ax=plt.subplots(figsize=(16,12))

df2=df1.groupby('City')['Total Rated Electric Generation (kW)'].sum().reset_index().set_index('City')

df3=df2.sort_values(['Total Rated Electric Generation (kW)'],ascending=False)[:15]

sns.barplot(x=df3.index,y=df3['Total Rated Electric Generation (kW)'])

for p in ax.patches:

    ax.annotate(format(round(p.get_height(),2)), (p.get_x()+0.05, p.get_height()+3))

    ax.set_title('Top 15 Cities with highest rated electric Generation',fontsize=25)



plt.xticks(rotation=45)

plt.show()    





#Cities with lowest power consmption

f,ax=plt.subplots(figsize=(16,12))

df2=df1.groupby('City')['Total Rated Electric Generation (kW)'].sum().reset_index().set_index('City')

df3=df2.sort_values(['Total Rated Electric Generation (kW)'],ascending=True)[64:75]

sns.barplot(x=df3.index,y=df3['Total Rated Electric Generation (kW)'])

for p in ax.patches:

    ax.annotate(format(round(p.get_height(),2)), (p.get_x()+0.05, p.get_height()+3))

    ax.set_title('Bottom 10 Cities with Lowest rated electric Generation',fontsize=25)



plt.xticks(rotation=45)

plt.show()    

#Power consumption by categories and subcategories

df3=df1.groupby(['Category','Subcategory'])['Total Rated Electric Generation (kW)'].mean().reset_index().set_index('Category')[:20]



df3.columns=['Subcategory','Avg Power consumption by categories']





df3
#Total power consption by each categories

df4=df1.groupby(['Category'])['Total Rated Electric Generation (kW)'].sum().reset_index().set_index('Category')

df4.sort_values('Total Rated Electric Generation (kW)',ascending=True)[:20]

df4.columns=['Total power consumption per categories']

df4

#Visualizing the categorical consmption of power

f,ax=plt.subplots(figsize=(18,12))

sns.barplot(x=df4.index,y=df4['Total power consumption per categories'])

for p in ax.patches:

    ax.annotate(format(round(p.get_height(),2)), (p.get_x()+0.05, p.get_height()+9))

    ax.set_title('Categorical wise power consumption',fontsize=25)



plt.xticks(rotation=45)

plt.show()    

#top 10 weather stations

plt.subplots(figsize=(18,12))

df1['Source Weather Station for Ambient Temperature Data'].value_counts()[:10].plot(kind='bar')

plt.title('Top 10 Weather stations',fontsize=25)
#city wise facility counts

df2=df1.groupby('City')['Facility Name'].count().reset_index().set_index('City')

df3=df2.sort_values(['Facility Name'],ascending=False)[:20]

df3
#City wise facility count



plt.subplots(figsize=(18,12))

sns.barplot(x=df3.index,y=df3['Facility Name'])

for p in ax.patches:

    ax.annotate(format(round(p.get_height(),2)), (p.get_x()+0.05, p.get_height()+9))

    ax.set_title('No of facilities per City',fontsize=20)



plt.xticks(rotation=45)

plt.title('City wise facility count',fontsize=20)

plt.show()    

df1.head()
df5=df1.groupby('City')["Longitude (°E)", "Latitude (°N)"].mean()

df6=df1.groupby('City')['Total Rated Electric Generation (kW)'].mean()

df65=df1['State']

df7=pd.concat([df5,df6],axis=1)

df8=df7.sort_values(['Total Rated Electric Generation (kW)'],ascending=False)[1:100]



df8.head()
df8['City']=df8.index

df8.head(3)





plt.subplots(figsize=(16,12))

sns.scatterplot( x=df8["Longitude (°E)"], y=df8["Latitude (°N)"],s=df8['Total Rated Electric Generation (kW)'])

plt.title('Area wise electric power generation in New York',fontsize=25)

plt.xlabel('Longitude of new york',fontsize=15)

plt.ylabel('latitude of New york',fontsize=15)

plt.show()

df1.describe()
df1[df1['Total Rated Heat Generation (MBtu/h)']==127000]
df1[df1['Total Rated Electrical Discharge Capacity (kW)']==20000]

df1[df1['Total Rated Electrical Storage Capacity (kWh)']==40000]
df1[df1['Total Rated Cooling Energy Storage Capacity (ton-hour)']==30000]
df77=df1.groupby('City')['Latitude (°N)','Longitude (°E)','Total Rated Electric Generation (kW)'].mean().reset_index()

df77.head()




import webbrowser

import folium

from IPython.display import display



USA_COORDINATES = (43.0902 , -76.7129  )



# create map of New York using latitude and longitude values

myMap1 = folium.Map(location=USA_COORDINATES, zoom_start=7)



myMap1 = folium.Map(location=USA_COORDINATES, zoom_start=8)



for lat, lon, city ,energy in zip(df77['Latitude (°N)'], df77['Longitude (°E)'],df77['City'],df77['Total Rated Electric Generation (kW)']):

    folium.Marker(

        [lat, lon],

       

        popup = ('City: ' + str(city).capitalize()+ '<br>'

                 'Energy: ' + str(energy)

                )

       

        ).add_to(myMap1)

myMap1




