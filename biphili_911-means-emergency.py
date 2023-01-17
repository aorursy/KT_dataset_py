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
import matplotlib.pyplot as plt

import seaborn as sns

#plt.style.use('fivethirtyeight')

import warnings

warnings.filterwarnings('ignore')
df=pd.read_csv('../input/montcoalert/911.csv')

df.head()
print('Rows     :',df.shape[0])

print('Columns  :',df.shape[1])

print('\nFeatures :\n     :',df.columns.tolist())

print('\nMissing values    :',df.isnull().values.sum())

print('\nUnique values :  \n',df.nunique())
df['twp'].values
df.index
df['lat'].dtype
df['zip'].value_counts().head(5).plot.bar();

plt.xlabel('Zip Code')

plt.ylabel('Count')

plt.show()
df['twp'].value_counts().head(5).plot.bar();

plt.xlabel('Township')

plt.ylabel('Count')

plt.show()
len(df['title'].unique())
df['title'].nunique()
x=df['title'].iloc[0]
x.split(':')[0]
df['Reason']=df['title'].apply(lambda x:x.split(':')[0])

df['Reason'].unique()
f,ax=plt.subplots(1,2,figsize=(18,8))

df['Reason'].value_counts().plot.pie(explode=[0,0.1,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)

ax[0].set_title('Reason for Call')

ax[0].set_ylabel('Count')

sns.countplot('Reason',data=df,ax=ax[1],order=df['Reason'].value_counts().index)

ax[1].set_title('Count of Reason')

plt.show()
type(df['timeStamp'].iloc[0])
df['timeStamp']=pd.to_datetime(df['timeStamp'])
type(df['timeStamp'].iloc[0])
time=df['timeStamp'].iloc[0]

time.hour
time.year
time.month
time.dayofweek
df['Hour']=df['timeStamp'].apply(lambda x:x.hour)

df['Month']=df['timeStamp'].apply(lambda x:x.month)

df['DayOfWeek']=df['timeStamp'].apply(lambda x:x.dayofweek)
dmap={0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
byMonth=df.groupby('Month').count()

byMonth['lat'].plot();
sns.lmplot(x='Month',y='twp',data=byMonth.reset_index());
mmap={1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
df['Month']=df['Month'].map(mmap)
df['DayOfWeek']=df['DayOfWeek'].map(dmap)
df.head()
sns.set_style('darkgrid')

f,ax=plt.subplots(1,2,figsize=(18,8))

k1=sns.countplot(x='DayOfWeek',data=df,ax=ax[0],palette='viridis')

k2=sns.countplot(x='DayOfWeek',data=df,hue='Reason',ax=ax[1],palette='viridis')
sns.set_style('darkgrid')

f,ax=plt.subplots(1,2,figsize=(18,8))

k1=sns.countplot(x='Month',data=df,ax=ax[0],palette='viridis')

k2=sns.countplot(x='Month',data=df,hue='Reason',ax=ax[1],palette='viridis')
df['Date']=df['timeStamp'].apply(lambda x:x.date())
#df.head()
plt.figure(figsize=(20,10))

df.groupby('Date').count()['lat'].plot();

plt.tight_layout()
plt.figure(figsize=(20,10))

df[df['Reason']=='Traffic'].groupby('Date').count()['lat'].plot();

plt.title('Calls Per Day for Traffic Issues');

plt.tight_layout()
plt.figure(figsize=(20,10))

df[df['Reason']=='Fire'].groupby('Date').count()['lat'].plot();

plt.title('Calls Per Day for Fire Issues');

plt.tight_layout()
plt.figure(figsize=(20,10))

df[df['Reason']=='EMS'].groupby('Date').count()['lat'].plot();

plt.title('Calls Per Day for EMS Issues');

plt.tight_layout()
dayHour=df.groupby(by=['DayOfWeek','Hour']).count()['Reason'].unstack()
plt.figure(figsize=(12,6))

sns.heatmap(dayHour,cmap='viridis');
plt.figure(figsize=(12,6));

sns.clustermap(dayHour,cmap='viridis');
dayMonth=df.groupby(by=['DayOfWeek','Month']).count()['Reason'].unstack()
plt.figure(figsize=(12,6))

sns.heatmap(dayMonth,cmap='viridis');
plt.figure(figsize=(12,6));

sns.clustermap(dayMonth,cmap='coolwarm');