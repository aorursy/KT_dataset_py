# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd
import numpy as np
import matplotlib.pyplot as mlt
import seaborn as sns
sns.set()
%matplotlib inline
data = pd.read_csv('../input/Attacks on Political Leaders in Pakistan.csv',na_values=' ',encoding='unicode_escape')
df= data.copy()
df
df.columns
df.dtypes
df.describe()
df.describe(include='all')
df.isnull().sum()
df[df['Location Category'].isnull()]
df[['City','Space (Open/Closed)']]
df.loc[df[df['Location Category'].isnull()].index ,'Location Category'] = 'UNKNOWN'
df['Location Category'].unique()
df['Location Category'].isnull().sum()
df.isnull().sum()
df
sns.countplot(x='Target Status',hue='Target Status',data=df);
sns.countplot(x='Province',hue='Target Status',data=df);
sns.countplot(x='Space (Open/Closed)',hue='Target Status',data=df);
mlt.figure(figsize=(50,10))
sns.countplot(x='Party',hue='Target Status', data=df)
sns.countplot(x='Day',hue='Target Status', data=df);
df['City'].unique()
#df.drop(['Total Killed','Total Injured','Total Killed by Day','Total Injured by Day'],axis = 1,inplace=True)

df['Total Killed'] = '0'
df['Total Killed'] = df['Total Killed'].astype(np.int32)

for x in df['City'].unique():
    tt = df[df['City']==x]
    df.loc[df['City']==x, 'Total Killed'] = tt['Killed'].sum()
    
df['Total Injured'] = '0'
df['Total Injured'] = df['Total Injured'].astype(np.int32)


for x in df['City'].unique():
    tt = df[df['City']==x]
    df.loc[df['City']==x, 'Total Injured'] = tt['Injured'].sum()
    


df['Total Killed by Day'] = '0'
df['Total Killed by Day'] = df['Total Killed by Day'].astype(np.int32)

for x in df['Day'].unique():
    tt = df[df['Day']==x]
    df.loc[(df['Day']==x), 'Total Killed by Day'] = tt['Killed'].sum()
        
    
    
df['Total Injured by Day'] = '0'
df['Total Injured by Day'] = df['Total Injured by Day'].astype(np.int32)

for x in df['Day'].unique():
    tt = df[df['Day']==x]    
    df.loc[(df['Day']==x) , 'Total Injured by Day'] = tt['Injured'].sum()
df
mlt.figure(figsize=(30,6))
sns.countplot(x='City',hue='Total Killed', data=df);
mlt.figure(figsize=(15,6))
sns.countplot(x='Day',hue='Total Killed by Day', data=df);
mlt.figure(figsize=(15,6))
sns.countplot(x='Day',hue='Total Injured by Day', data=df);
df['Total Killed by Time'] = '0'
df['Total Killed by Time'] = df['Total Killed by Time'].astype(np.int32)

for x in df['Time'].unique():
    tt = df[df['Time']==x]
    df.loc[(df['Time']==x), 'Total Killed by Time'] = tt['Killed'].sum()
df
mlt.figure(figsize=(15,6))
sns.barplot(data= df.groupby('Time')['Total Killed by Time'].value_counts(normalize=True).reset_index(name='Perc') ,x='Time',y='Perc',hue='Total Killed by Time',dodge = True)
df['Total Killed by Day Type'] = '0'
df['Total Killed by Day Type'] = df['Total Killed by Day Type'].astype(np.int32)

for x in df['Day Type'].unique():
    tt = df[df['Day Type']==x]
    df.loc[(df['Day Type']==x), 'Total Killed by Day Type'] = tt['Killed'].sum()
sns.countplot(x='Day Type' ,hue='Total Killed by Day Type',data=df )
mlt.figure(figsize=(10,6))
sns.barplot(data= df.groupby('Day Type')['Total Killed by Day Type'].value_counts(normalize=True).reset_index(name='Perc') ,x='Day Type',y='Perc',hue='Total Killed by Day Type',dodge = True)
sns.barplot(data= df.groupby('Space (Open/Closed)')['Target Status'].value_counts(normalize=True).reset_index(name='Perc') ,x='Space (Open/Closed)',y='Perc',hue='Target Status',dodge = True)
sns.barplot(data= df.groupby('Day')['Target Status'].value_counts(normalize=True).reset_index(name='Perc') ,x='Day',y='Perc',hue='Target Status',dodge = True)
temp = df.groupby('City')['Target Status'].value_counts(normalize=True).reset_index(name='Perc')
temp
mlt.figure(figsize=(50,6));

mlt.subplot(121)
sns.barplot(data=temp,x='City',y='Perc',hue='Target Status',dodge = True)
df['Date']
df['Date']= pd.to_datetime(df['Date'],errors='coerce',dayfirst=True)
df
df['Month'] = pd.DatetimeIndex(df['Date']).month
df
mlt.figure(figsize=(15,6))
sns.countplot(x='Month',hue='Target Status', data=df);
df['Year'] = pd.DatetimeIndex(df['Date']).year
df
mlt.figure(figsize=(15,6))
sns.countplot(x='Year',hue='Target Status', data=df);
df


kk = df[['Year','Killed','Month']]

kk['By Year'] = '0'
kk['By Year'] = kk['By Year'].astype(np.int32)

for x in kk['Year'].unique():
    kk.loc[kk['Year']==x,'By Year'] = kk.loc[kk['Year']==x,'Killed'].sum()
    
kk['By Month'] = '0'
kk['By Month'] = kk['By Month'].astype(np.int32)

for x in kk['Month'].unique():
    kk.loc[kk['Month']==x,'By Month'] = kk.loc[kk['Month']==x,'Killed'].sum()
kk
mlt.figure(figsize=(15,6))
sns.countplot(x='Year',hue='By Year', data=kk);
mlt.figure(figsize=(15,6))
sns.countplot(x='Month',hue='By Month', data=kk);
sns.pairplot(df);
cor = df.corr()
mlt.figure(figsize=(10,6))
sns.heatmap(cor, annot=True);
