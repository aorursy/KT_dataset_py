# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
flights_df=pd.read_csv('../input/2008.csv')
flights_df.head()
flights_df.columns
flights_df['DepHour'] = flights_df['DepTime'] // 100

flights_df['DepHour'].replace(to_replace=24, value=0, inplace=True)
flights_df.head()
flights_df['UniqueCarrier'].nunique()
list1=flights_df['Cancelled'].value_counts()

Answer= list1[0]-list1[1]

Answer
flights_df.groupby(by=['ArrDelay','DepDelay'])['Dest'].max()

#flights_df.loc[flights_df['ArrDelay'].max() and flights_df['DepDelay'].max()]
Cancelled_flights_of_each_carrier=flights_df.groupby(['UniqueCarrier','Cancelled'])

Cancelled_flights_of_each_carrier.size()
pd.crosstab(flights_df['DepHour'],flights_df['UniqueCarrier'],margins=True)

Least_percentage_of_cancelled_flights=flights_df.groupby(['Cancelled','DepHour'])

Least_percentage_of_cancelled_flights.size()

#pd.pivot_table(flights_df,index=['DepHour','Cancelled'],columns='UniqueCarrier')

#pd.crosstab(Carrier_hour,Cancelled_flights_of_each_carrier,margins=True)
flights_df[(flights_df['DepDelay']< 0) & (flights_df['Cancelled'] == 0)]['DepHour'].value_counts()
flights_df[flights_df['Cancelled']==0]['DepHour'].value_counts()

#Cancelled_flights_of_each_carrier=flights_df.groupby(['UniqueCarrier','Cancelled'])

#Cancelled_flights_of_each_carrier.size()
#Top_10 = flights_df.groupby(['UniqueCarrier','Cancelled'])

#Top_10.size()

flights_df[flights_df['Cancelled']==0]['UniqueCarrier'].value_counts()
flights_df['CancellationCode'].value_counts()
flights_df['route']=flights_df['Origin'] + flights_df['Dest']
flights_df['route'].value_counts()
#List1={}

#List1=flights_df[ (flights_df['DepDelay'] > 0)]['route'].value_counts()

#List2=List1[:5].keys()

DataFrame_crosstab=pd.crosstab(index=[flights_df['DepDelay'] > 0,flights_df['CancellationCode']],columns=(flights_df['route']),margins=True)

Table=DataFrame_crosstab.iloc[4:8,:]

Table2=Table.sort_values(by=['All'][:],axis=0,ascending=True).head().T

Table2.sort_values(by=['All'],ascending=False)

flights_df['DepHour'].plot.hist()
flights_df[flights_df['CancellationCode'] == 'A']['Month'].value_counts()
flights_df[(flights_df['CancellationCode'] == 'A') & (flights_df['Month'] == 4)]['UniqueCarrier'].value_counts()