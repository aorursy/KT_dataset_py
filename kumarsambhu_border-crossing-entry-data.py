# 1.0 Reset memory

#     ipython magic command

%reset -f



import pandas as pd

import numpy as np

import os



# 1.1 For chi-square tests

from scipy.stats import chi2_contingency

# 1.2 For t-test

from scipy.stats import ttest_ind

# 1.3 Finding out score at a percentile point and

#     pearson correlation coeff function

from scipy.stats import scoreatpercentile, pearsonr



# 1.4

import matplotlib.pyplot as plt

import matplotlib

import matplotlib as mpl     # For creating colormaps



import seaborn as sns

# 1.4.1 Mosaic plots

# https://www.statsmodels.org/dev/graphics.html

# https://www.statsmodels.org/dev/generated/statsmodels.graphics.mosaicplot.mosaic.html#statsmodels.graphics.mosaicplot.mosaic

from statsmodels.graphics.mosaicplot import mosaic





# 1.4 Misc facilities

from collections import Counter

import os, time, sys, gc



# 1.5 For data processing

# 1.5.1 Scale data

from sklearn.preprocessing import StandardScaler

# 1.5.2 Split dataset

from sklearn.model_selection import train_test_split

# 1.5.3 Class to develop kmeans model

from sklearn.cluster import KMeans



# 1.5 Display as many columns as possible

pd.set_option('display.max_columns', 500)

# os.chdir("C:\\mks\\eCBDADS\\Sessions")
# os.listdir()
df = pd.read_csv("../input/border-crossing-entry-data/Border_Crossing_Entry_Data.csv")
gc.collect()
# 2.2 Explore data

df.shape
df.head()
df.columns
df.columns.values
df.values
# 2.3

sys.getsizeof(df)
np.sum(df.memory_usage())
# 2.4 Data types

df.dtypes
New_Columns={

             'Port Name' : 'port_name',

             'State'     : 'state',

             'Port Code' : 'port_code',

             'Border'    : 'border',

             'Date'      : 'date',

             'Measure'   : 'measure',

             'Value'     : 'value',

             'Location'  : 'location'

             }

df.rename(New_Columns,inplace='True',axis=1)
df.columns.values
df['location'] = df['location'].astype('category')
df['measure'] = df['measure'].astype('category')
df['port_code'] = df['port_code'].astype('category')
gc.collect()
# 3.2 So what is our file size now?

sys.getsizeof(df)
np.sum(df.memory_usage())
# 4. Conversion of datetime to 'datetime' datatype.

 #    This will further save memory.

 #    'object' is most general type of datatype

 #     Transforming to known types, saves space



df['date'] = pd.to_datetime(df['date'])
df['date'].dtypes.name
# 4.1 What is the filesize now?

sys.getsizeof(df)
gc.collect()
np.sum(df.memory_usage())
# 4.2 Extract year, month, day etc

df['date'].dt.year
df['date'].dt.month
df['date'].dt.day
# 5 (Q1) How many unique port_name, state & measures exist

df['port_name'].nunique()
df['port_name'].value_counts().values
df['port_name'].value_counts().shape
df['state'].value_counts().values
df['measure'].value_counts().values
####### Groupby:



# 5.1 (Q2) Where are the most passengers crossing at which port_name?

result = df.groupby('port_name')['value'].min().sort_values(ascending=False)
type(result) 
result.head()
result.tail()
result.size
# 5.2 (Q3) Recency: Find the last border cross date at each port

result1 = df.groupby('port_name')['date'].max().sort_values(ascending = False)
result1.head()
result1.size
# 6.  (Q5): What is total no of passengers during the period of data

passengers=['Personal Vehicle Passengers','Personal Vehicles','Pedestrians','Train Passengers','Bus Passengers']
df.loc[df['measure'].isin(passengers),'type'] ='passengers'
df.loc[~df['measure'].isin(passengers),'type'] ='vehicles'
df['type'].unique()
plt.figure(figsize=(25,10))
sns.barplot(data=df[df['type']=='passengers'],x='border',y='value', estimator =sum)
# 6.  (Q6): What are total vehicles travelled accross bonders



sns.barplot(data=df[df['type']=='vehicles'], x='border',y='value',estimator = sum)
sns.barplot(data=df[df['type']=='vehicles'], x='border',y='value', order= ['US-Canada Border','US-Mexico Border'],estimator = sum)
plt.figure(figsize=(15,3))



sns.barplot(data=df[df['type']=='vehicles'], x='measure',y='value',hue='border',estimator =sum)
plt.figure(figsize=(15,3))



sns.barplot(data=df[df['type']=='passengers'], x='measure',y='value',hue='border',estimator =sum)
#7) find sum of values for all measures



df.groupby('measure').sum().value
#8) Use group by method to find sum of values for all passengers for borders

all1= df[df.type=='passengers'].groupby(['border','measure']).sum().value.reset_index()



all1
plt.figure(figsize=(15,3))

sns.barplot(data=all1, x='measure',y='value',hue='border')
#8) which state in US has max number and lowest number of passengers corssed 



stateData=df.groupby(['state','type'])['value'].sum().sort_values(ascending=False).reset_index()
#8) which state in US has max number and lowest number of passengers corssed 



plt.figure(figsize=(15,3))

sns.barplot(data=stateData[stateData.type=='passengers'],x='state',y='value')
#8) which state has highest and lowest numebr vehicles travelled



StateVehicle=df.groupby(['state','type'])['value'].sum().sort_values(ascending=False).reset_index()
#8) which state has highest and lowest numebr vehicles travelled



plt.figure(figsize=(15,3))



sns.barplot(data=StateVehicle[StateVehicle.type=='vehicles'],y = 'state', x= 'value')
portdata=df.groupby(['port_name','type'])['value'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(15,40))



sns.barplot(data=portdata[portdata.type=='passengers'],y='port_name',x='value')
#12) which year has highest entries of passengers and vehicle across borders



df.date=pd.to_datetime(df.date)



import datetime

df['year']=pd.DatetimeIndex(df['date']).year



df['month']=pd.DatetimeIndex(df['date']).month
yearlyD=df.groupby('year')['value'].mean().sort_values(ascending = False).reset_index()
plt.figure(figsize =(15,10))



ax = sns.barplot(x = yearlyD.year, y=yearlyD.value,data=yearlyD)



xticks= ax.set_xticklabels(ax.get_xticklabels(), rotation = 30)