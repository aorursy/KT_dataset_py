import numpy as np

import pandas as pd
from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
#Loading Data

df = pd.read_csv("../input/IndiaAffectedWaterQualityAreas.csv",encoding='latin1')

df.head()
df.columns
df.shape
df.dtypes
df.isnull().sum()
#Checking for Duplicates

Duplicates=df[df.duplicated(keep=False)]

Duplicates.shape
#Dropping Duplicates

df1=df.drop_duplicates(subset=['State Name', 'District Name', 'Block Name', 'Panchayat Name',

       'Village Name', 'Habitation Name', 'Quality Parameter', 'Year'], keep=False)

df1.shape
df1.describe()
df1['Quality Parameter'].value_counts()
import seaborn as sns

sns.countplot(df1['Quality Parameter'])
df1['State Name'].value_counts()

#This may be because area wise Rajasthan is ranked highest in India
df1['Quality Parameter'].groupby(df1['State Name']).describe()
#Splitting the Year column to retain just the year

df1['year'] = pd.DatetimeIndex(df1['Year']).year

df1=df1.drop(columns='Year')
#Subsetting the data

df1_new=df1.loc[df['State Name'].isin(['RAJASTHAN','BIHAR','ASSAM','ORISSA'])] 

Subset_Data = df1_new[['State Name', 'Quality Parameter', 'year']]
#Assigning a numerical value to all the Quality Parameters

import sklearn

from sklearn.preprocessing import LabelEncoder

numbers = LabelEncoder()

Subset_Data['Quality'] = numbers.fit_transform(Subset_Data['Quality Parameter'].astype('str'))
State_Quality_Count = pd.DataFrame({'count' : Subset_Data.groupby( [ "State Name", "Quality","Quality Parameter"] ).size()}).reset_index()

State_Quality_Count.head()
High_Quality_count = State_Quality_Count.sort_values(['count'], ascending=[False])

High_Quality_count.head()
State_Quality_Count_year = pd.DataFrame({'count' : Subset_Data.groupby( [ "State Name", "Quality","Quality Parameter","year"] ).size()}).reset_index()

State_Quality_Count_year
State_Quality_Count_year['rank']=State_Quality_Count_year.groupby(['State Name','Quality'])['count'].rank("dense", ascending=False)

State_Quality_Count_year.head()
Top_count=State_Quality_Count_year[State_Quality_Count_year['rank']==1]
import matplotlib.pyplot as plt

freq_plot = Top_count['year'].value_counts().plot(kind='bar',figsize=(9,5),title="Year with highest water degradation")

freq_plot.set_xlabel("Year")

freq_plot.set_ylabel("Frequency")

plt.show()
Subset_Data2 = df1[['State Name', 'Quality Parameter', 'year']]

Subset_Data2['Quality'] = numbers.fit_transform(Subset_Data2['Quality Parameter'].astype('str'))

SQT = pd.DataFrame({'count' : Subset_Data2.groupby( [ "State Name", "Quality","Quality Parameter","year"] ).size()}).reset_index()

SQT['rank']=SQT.groupby(['State Name','Quality'])['count'].rank("dense", ascending=False)

Top_count2=SQT[SQT['rank']==1]
freq_plot = Top_count2['year'].value_counts().plot(kind='bar',figsize=(9,5),title="Year with highest water degradation")

freq_plot.set_xlabel("Year")

freq_plot.set_ylabel("Frequency")

plt.show()
Quality = pd.DataFrame({'count' : Subset_Data2.groupby( [ "Quality","year"] ).size()}).reset_index()

Quality.head()
Arsenic=Quality[Quality['Quality']==0]

Arsenic.plot('year','count',kind='scatter', layout=(5,5),title= "Arsenic distribution")

plt.show()
Fluoride=Quality[Quality['Quality']==1]

Fluoride.plot("year",'count',kind='scatter', layout=(5,5),title= "Fluoride distribution")

plt.show()
Iron=Quality[Quality['Quality']==2]

Iron.plot('year','count',kind='scatter', layout=(5,5),title= "Iron distribution")

plt.show()
Nitrate=Quality[Quality['Quality']==3]

Nitrate.plot('year','count',kind='scatter', layout=(5,5),title= "Nitrate distribution")

plt.show()
Salinity=Quality[Quality['Quality']==4]

Salinity.plot('year','count',kind='scatter', layout=(5,5),title= "Salinity distribution")

plt.show()
State_Quality = pd.DataFrame({'count' : Subset_Data2.groupby( [ "State Name","Quality","year"] ).size()}).reset_index()

State_Nitrate_Quality=State_Quality[(State_Quality['year']==2011) & (State_Quality['Quality']==3)]

State_Nitrate_Quality.sort_values(['count'], ascending=[False])
State_Nitrate_Quality1=State_Quality[(State_Quality['year']==2010) & (State_Quality['Quality']==3)]

State_Nitrate_Quality1.sort_values(['count'], ascending=[False])