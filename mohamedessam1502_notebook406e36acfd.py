import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
#Import the emergency file
emer=pd.read_csv('../input/montcoalert/911.csv')
emer.head()
emer.info()
emer.describe()
emer.info()
#Time stamp us not an object type
#Alter timestamp
emer['timeStamp']=pd.to_datetime(emer['timeStamp'])
emer['timeStamp'].dtypes
#check if there is Null values
emer.isna().sum()
#Fill na
Nan_values=['zip','twp','addr']
i=0
while(i<3):
    emer[Nan_values[i]].fillna(emer[Nan_values[i]].mode()[0],inplace=True)
    i=i+1
emer.info()
emer['zip'].unique()
#Alter zip code type
emer['zip']=emer['zip'].astype('int64')
emer.info()
#Drop duplicates if there is any
emer.drop_duplicates()
#Split time stamp into different columns
emer['day']=emer['timeStamp'].dt.date
emer['time']=emer['timeStamp'].dt.time
#get station column
emer['station']=emer['desc'].str.split(';',expand=True)[:][2].str.split('-',expand=True)[0].str.split('20',expand=True)[0]
#drop timeStamp label
emer.drop(columns=['timeStamp'],axis=1,inplace=True)
#drop describtion label
emer.drop(columns=['desc'],axis=1,inplace=True)
emer.info()
#split dataframe for day and night
import datetime as dt
emer_day=emer[emer['time']<dt.time(13)]
emer_night=emer[emer['time']>=dt.time(13)]
emer.head()
emer=emer[['lat','lng','zip','title','twp','addr','station','day','time','e']]
emer.head()
#show number of emergency an non using count plot
sns.countplot(x=emer['e'])
#Seems like they are all emergency... Do not know what to predict
sns.set(rc={'figure.figsize': [7, 7]}, font_scale=1.2)
#Which station has the most emergency
sns.set(rc={'figure.figsize': [12, 12]}, font_scale=1.2)
sns.countplot(y=emer['station'])
emer[emer['station']==' '].count()
#seems like many of station values are blank which is more than half of the dataset so we cannot depend on the station to
#show relations
emer['twp'].value_counts()
#the relation between twp and emergency title
twp_title=emer.loc[:,['twp','title']]
emer.info()
#The town with most emergency rate
sns.set(rc={'figure.figsize': [12, 12]}, font_scale=1.2)
sns.barplot(x=emer['twp'].value_counts().head(5).index,y=emer['twp'].value_counts().head(5))
#Lower merion is the town with the most rate of emergencies
emer.head()
#slice dataset for emergencies of lower merion city
emer_lower_merion=emer[emer['twp']=='LOWER MERION']
emer_lower_merion['title'].nunique()
sns.barplot(x=emer_lower_merion['title'].value_counts().head(3).index,y=emer_lower_merion['title'].value_counts().head(3))
#the most 3 common emergencies in lower merion is:
#1.vechile accidents
#2.disabled or broken vechile
#3.fire alarm
emer.head()
twp_title
#Im done of working on visualiztion with fuckin’ categorical data only
#all can be done with only pandas