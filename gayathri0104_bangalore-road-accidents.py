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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
input_data_dir = "../input/bangalore-accident-data"

input_data_file = "bangalore-cas-alerts.csv"

input_data_path = os.path.join(input_data_dir, input_data_file)

df = pd.read_csv(input_data_path)
df.head()
df.shape
df.dropna(inplace=True) 

# Dropping the missing values
df.isnull().sum()

# Checking for null values
df =df.rename(columns = {'deviceCode_deviceCode':'DeviceCode',

        'deviceCode_location_latitude':'Latitude',

        'deviceCode_location_longitude' : 'Longitude',

        'deviceCode_location_wardName':'WardName',

        'deviceCode_pyld_alarmType':'AlarmType',

        'deviceCode_pyld_speed':'Speed',

        'deviceCode_time_recordedTime_$date':'RecordedDateTime'})
df = df[~df.duplicated()]
df.shape
for i in df.columns:

    print(i,df[i].nunique())

    

# getting the unique values in the data
plt.figure(figsize=[15,6])

df.WardName.value_counts().plot(kind='bar')

plt.title('No. of Alarms in each ward')

plt.show()
# We have 2 wards names are other and Other. combining all as one

df.WardName = df.WardName.replace({'other':'Other'})
df.WardName.nunique()
Xtrain = df[['Latitude','Longitude']][df.WardName != 'Other']

ytrain = df.WardName[df.WardName != 'Other']
Xtest = df[['Latitude','Longitude']][df.WardName == 'Other']

ytest = df.WardName[df.WardName == 'Other']
Xtrain.shape, ytrain.shape, Xtest.shape, ytest.shape
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1,weights='distance')

knn.fit(Xtrain,ytrain)

ypred = knn.predict(Xtest)
df.WardName.value_counts().head()
df.WardName[df.WardName =='Other'] = ypred

# the results of knn which is in ypred is assigned to ward names categorized as Other
df.WardName.value_counts().head()
plt.figure(figsize=[15,6])

plt.subplot(1,2,1)

df.WardName.value_counts().head(10).plot(kind='bar')

plt.title('Most Dangerous Wards')



plt.subplot(1,2,2)

df.WardName.value_counts().tail(10).plot(kind='bar')

plt.title('Most Safe Wards')

plt.show()
df.AlarmType = df.AlarmType.replace({'UFCW':'FCW','LDWL':'LDW','LDWR':'LDW'})
df.AlarmType.value_counts()
plt.figure(figsize=[15,6])

df.AlarmType.value_counts().plot(kind='bar')

plt.show()
data = df.WardName.value_counts().head(10)

ward = data.index

ward_top = df[df.WardName.isin(ward)]
plt.figure(figsize=[15,6])

sns.countplot(x=ward_top.WardName,hue=ward_top.AlarmType)

plt.title('Distribution of Alarm Type in most dangerous wards')

plt.show()
plt.figure(figsize=[15,6])

sns.kdeplot(df.Speed,shade=True,color='y')

plt.axvline(df.Speed.mean(),linestyle='dashed',linewidth='2',color='k',label=df.Speed.mean())

plt.legend(loc='best')

plt.title('Average Speed of the buses from 6AM to 6PM')

plt.show()
fig, axes = plt.subplots(3, 2, figsize=(20,10))





sns.kdeplot(df.Speed[df.AlarmType=='PCW'],shade=True,ax=axes[0][0])

axes[0][0].axvline(df.Speed[df.AlarmType=='PCW'].mean(),linestyle='dashed',color='k',label='PCW '+str(df.Speed[df.AlarmType=='PCW'].mean()))

axes[0][0].legend(loc='best')







sns.kdeplot(df.Speed[df.AlarmType=='FCW'],color='g',shade=True,ax=axes[0][1])

axes[0][1].axvline(df.Speed[df.AlarmType=='FCW'].mean(),linestyle='dashed',color='k',label='FCW '+str(df.Speed[df.AlarmType=='FCW'].mean()))

axes[0][1].legend(loc='best')







sns.kdeplot(df.Speed[df.AlarmType=='Overspeed'],color='y',shade=True,ax=axes[1][0])

axes[1][0].axvline(df.Speed[df.AlarmType=='Overspeed'].mean(),linestyle='dashed',color='k',label='Overspeed '+str(df.Speed[df.AlarmType=='Overspeed'].mean()))

axes[1][0].legend(loc='best')





sns.kdeplot(df.Speed[df.AlarmType=='HMW'],color='cyan',shade=True,ax=axes[1][1])

axes[1][1].axvline(df.Speed[df.AlarmType=='HMW'].mean(),linestyle='dashed',color='k',label='HMW '+str(df.Speed[df.AlarmType=='HMW'].mean()))

axes[1][1].legend(loc='best')





sns.kdeplot(df.Speed[df.AlarmType=='LDW'],color='m',shade=True,ax=axes[2][0])

axes[2][0].axvline(df.Speed[df.AlarmType=='LDW'].mean(),linestyle='dashed',color='k',label='LDW '+str(df.Speed[df.AlarmType=='LDW'].mean()))

axes[2][0].legend(loc='best')

plt.show()
df.RecordedDateTime = df.RecordedDateTime.map(lambda x : pd.Timestamp(x, tz='Asia/Kolkata'))
df['Month'] = df.RecordedDateTime.dt.month_name()

df['Year'] = df.RecordedDateTime.dt.year

df['Date'] = df.RecordedDateTime.dt.day

df['Weekday'] = df.RecordedDateTime.dt.day_name()
df.Month.unique()
df.Year.unique()
df.Date.unique()
df['Hour'] = df.RecordedDateTime.dt.hour
df.Hour.unique()
plt.figure(figsize=[15,6])

df.Month.value_counts().plot(kind='bar')

plt.show()
plt.figure(figsize=[15,6])

df.Weekday.value_counts().plot(kind='bar')

plt.show()
hr = df.Hour.value_counts().sort_index()

hr.index
plt.figure(figsize=[15,6])

plt.bar(hr.index, hr.values)

plt.xticks(np.arange(1,25))

plt.title('No. of Alert on hourly basis')

plt.show()
df.groupby('WardName')['Hour'].value_counts().sort_values(ascending=False).head(50)
df.groupby('WardName')['AlarmType'].value_counts().sort_values(ascending=False).head(50)