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

%matplotlib inline

import seaborn as sns

sns.set(rc={'figure.figsize':(16,8)})
leads = pd.read_csv('/kaggle/input/telecaller-operations-dataset/leads.csv', encoding='latin1')

leads
leads['userId'].nunique()
leads1 = leads.drop(['id', 'userId', 'name', 'phoneNumber'], axis=1)

leads1
leads1.dtypes
leads1['createdAt'] = pd.to_datetime(leads1['createdAt'])

leads1['receivedAt'] = pd.to_datetime(leads1['receivedAt'])

leads1.isExternal = leads1.isExternal.astype(int)

leads1.dtypes
# leads1['city'].isnull().sum() ## 738 ## 7.44%

# leads1['city'].nunique() ## 61

leads1['city'].value_counts().head(10)
# leads1['state'].isnull().sum() ## 1380 ## 13.91%

# leads1['state'].nunique() ## 20

leads1['state'].value_counts().head(10)
# leads1['source'].isnull().sum() ## 2

# leads1['source'].nunique() ##108

leads1['source'].value_counts().head(20)
# leads1['isExternal'].isnull().sum() ## 0

# leads1['isExternal'].nunique() ##2

leads1['isExternal'].value_counts(normalize=True)*100
leads2 = leads1[['city', 'isExternal', 'source']].groupby(['city','isExternal']).count()

leads2 = leads2.reset_index()

leads2 = leads2.pivot(index='city', columns='isExternal', values='source')

del leads2.columns.name

leads2 = leads2.reset_index()

leads2.columns=['city', 'LeadsByJobAssist', 'LeadsByExternal']
leads20 = leads2.sort_values(by='LeadsByJobAssist', ascending=False)

leads20.reset_index(inplace=True)

# leads20.to_csv('leads20.csv', index=False)

leads20.head(10)
leads21 = leads2.sort_values(by='LeadsByExternal', ascending=False)

leads21.reset_index(inplace=True)

# leads21.to_csv('leads21.csv', index=False)

leads21.head(10)
leads3 = leads1[['state', 'isExternal', 'source']].groupby(['state','isExternal']).count()

leads3 = leads3.reset_index()

leads3 = leads3.pivot(index='state', columns='isExternal', values='source')

del leads3.columns.name

leads3 = leads3.reset_index()

leads3.columns=['state', 'LeadsByJobAssist', 'LeadsByExternal']
leads30 = leads3.sort_values(by='LeadsByJobAssist', ascending=False)

# leads30.to_csv('leads30.csv', index=False)

leads30.head(10)
leads31 = leads3.sort_values(by='LeadsByExternal', ascending=False)

# leads31.to_csv('leads31.csv', index=False)

leads31.head(10)
leads1['duration'] = pd.to_timedelta(leads1['createdAt'] - leads1['receivedAt'])

leads1['duration'] = leads1['duration'].dt.days

print(leads1['duration'].max())

print(leads1['duration'].min())

print(leads1['duration'].mean())

leads1
leads1.drop(['duration'], axis=1, inplace=True)

leads11 = leads1.copy()
leads11['createdAt_day'] = leads11['createdAt'].dt.day

leads11['createdAt_day'].value_counts()
leads11['createdAt_hour'] = leads11['createdAt'].dt.hour

leads11['createdAt_hour'].value_counts()
leads11['receivedAt_date'] = leads11['receivedAt'].dt.date

leads11['receivedAt_date'].value_counts().head(10)
leads11['receivedAt_date'].min()
leads11['receivedAt_date'].max()
leads11['receivedAt_dt_doy'] = pd.to_datetime(leads11['receivedAt'])

leads11['receivedAt_dt_doy'] = leads11['receivedAt_dt_doy'].dt.dayofyear

# leads11['receivedAt_dt_doy'].min() ## 43

# leads11['receivedAt_dt_doy'].max() ## 333
leads110 = leads11[leads11['isExternal']==0]

leads111 = leads11[leads11['isExternal']==1]
df_time110 = leads110[['receivedAt_dt_doy', 'isExternal']].groupby('receivedAt_dt_doy').count()

df_time110.reset_index(inplace=True)
df_time111 = leads111[['receivedAt_dt_doy', 'isExternal']].groupby('receivedAt_dt_doy').count()

df_time111.reset_index(inplace=True)
df_time11 = leads11[['receivedAt_dt_doy', 'isExternal']].groupby('receivedAt_dt_doy').count()

df_time11.reset_index(inplace=True)
df_time110_1 = df_time110[df_time110['isExternal']<151]

sns.lmplot(y='isExternal', x='receivedAt_dt_doy', data=df_time110_1)
df_time111_1 = df_time111[df_time111['isExternal']<75]

sns.lmplot(y='isExternal', x='receivedAt_dt_doy', data=df_time111_1)
df_time11_1 = df_time11[df_time11['isExternal']<201]

sns.lmplot(y='isExternal', x='receivedAt_dt_doy', data=df_time11_1)
telecallers = pd.read_csv('/kaggle/input/telecaller-operations-dataset/telecallers.csv', encoding='latin1')

telecallers
telecallers = telecallers.drop(['phoneNumber', 'createdAt'], axis=1)

telecallers.columns = ['telecallerId', 'telecallerName']

telecallers
lead_calls = pd.read_csv('/kaggle/input/telecaller-operations-dataset/lead_calls.csv', encoding='latin1')

lead_calls
lead_calls['id'].nunique()
lead_calls = pd.merge(telecallers, lead_calls, on='telecallerId', how='inner')

lead_calls = lead_calls.drop(['id', 'telecallerId'], axis=1)

lead_calls
lead_calls['telecallerName'].value_counts()
lead_calls['createdAt'] = pd.to_datetime(lead_calls['createdAt'])

lead_calls['calledAt'] = pd.to_datetime(lead_calls['calledAt'])
lead_calls['createdAt_dt_date'] = lead_calls['createdAt'].dt.date

lead_calls['createdAt_dt_date'].value_counts()
lead_calls['createdAt_dt_hour'] = lead_calls['createdAt'].dt.hour

lead_calls['createdAt_dt_hour'].value_counts()
lead_calls.drop(['createdAt_dt_date', 'createdAt_dt_hour', 'createdAt'], axis=1, inplace=True)
lead_calls['calledAt_dt_date'] = lead_calls['calledAt'].dt.date

lead_calls['calledAt_dt_date'].value_counts().head(10)
lead_calls.drop(['calledAt'], axis=1, inplace=True)
lead_calls['calledAt_dt_date'].min()
import datetime

lead_calls[lead_calls['calledAt_dt_date']>datetime.date(2016, 8, 26)]['calledAt_dt_date'].min()
lead_calls['calledAt_dt_date'].max()
# lead_calls['leadId'].value_counts().value_counts()

lead_calls['leadId'].value_counts().value_counts(normalize=True)*100
lead_calls