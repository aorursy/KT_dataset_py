# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# import the important libraries

import pandas as pd     # for dataframe

import numpy as np      # for arraies

import matplotlib.pyplot as plt  # for visualization 

%matplotlib inline

import seaborn as sns           # for visualization 
# load the data

no_show = pd.read_csv('/kaggle/input/noshowappointments/KaggleV2-May-2016.csv')

no_show.head()
no_show.shape
no_show.info()
no_show.describe() # for knowing statistical information
no_show.nunique()
no_show['Handcap'].unique()
no_show['Handcap'].value_counts()
# there are some anomalous data in Handcap column
no_show['Age'].value_counts().head()
no_show['Age'].value_counts().tail()
# Print Unique Values for 'Age'

np.sort(no_show['Age'].unique())
no_show.info()
no_show['Handcap'].replace([2,3,4],np.nan,inplace=True)
# Convert PatientId from Float to Integer

no_show['PatientId'] = no_show['PatientId'].astype('int64')



# Convert ScheduledDay and AppointmentDay from 'object' type to 'datetime64[ns]'

no_show['ScheduledDay'] = pd.to_datetime(no_show['ScheduledDay']).dt.date.astype('datetime64[ns]')

no_show['AppointmentDay'] = pd.to_datetime(no_show['AppointmentDay']).dt.date.astype('datetime64[ns]')



# Rename incorrect column names.

no_show = no_show.rename(columns={'Hipertension': 'Hypertension', 'Handcap': 'Handicap', 'SMS_received': 'SMSReceived', 'No-show': 'NoShow'})
# drop the nan in data

no_show.dropna(inplace=True)
# drop the Age which is less than 0

no_show = no_show[no_show["Age"]>=0]
no_show.head()
np.sort(no_show.AppointmentDay.unique())
np.sort(no_show.Neighbourhood.unique())
no_show.Neighbourhood.value_counts().head()
AppointmentDay = no_show.AppointmentDay.value_counts()

AppointmentDay
pd.DataFrame(AppointmentDay).plot(kind='bar',figsize=(16,6))

plt.title('Number of Appointments in each day')

plt.show()
no_show.head()
sns.countplot(no_show.Gender)

plt.title('Male vs Female')

plt.show()
sns.countplot(no_show.Gender,hue=no_show.NoShow)

plt.title('The number of men attending compared to women')

plt.show()
no_show.Age.plot(kind='hist',bins =50)

plt.title('distribution of Ages')

plt.show()
no_show.Age[no_show.Gender=='F'].value_counts().head(90).plot(kind='bar',figsize=(20,6))

plt.title(' top 90 frequanced ages of females')

plt.show()
no_show.Age[no_show.Gender=='M'].value_counts().head(90).plot(kind='bar',figsize=(20,6))

plt.title(' top 90 frequanced ages of males')

plt.show()
plt.figure(figsize=(20,6))

plt.xticks(rotation=90)

ax = sns.countplot(x=np.sort(no_show.Age))

ax.set_title("Number of Appointments by Age")

plt.show()
no_show.head()
plt.figure(figsize=(20,6))

plt.xticks(rotation=90)

sns.countplot(x=np.sort(no_show.Neighbourhood))

plt.title("Nnumber of Appointments by Neighbourhood")

plt.show()
l=list(no_show.Neighbourhood.value_counts().index)
plt.figure(figsize=(20,6))

plt.xticks(rotation=90)

sns.countplot(no_show.Neighbourhood,hue=no_show.NoShow)

plt.title("Show/NoShow by Neighbourhood")

plt.show()
plt.figure(figsize=(20,6))

plt.xticks(rotation=90)

sns.countplot(no_show.Neighbourhood,hue=no_show.SMSReceived)

plt.title('Neighbourhood which receive SMS')

plt.show()
no_show.Scholarship.value_counts()
sns.countplot(no_show.Scholarship,hue=no_show.NoShow)

plt.title('number of Patients attended on Appointment Day ')

plt.show()
sns.countplot(no_show.SMSReceived,hue=no_show.NoShow)

plt.title('number of Patients attended on Appointment Day ')

plt.show()
no_show.SMSReceived.value_counts()
no_show.info()
disease_data = no_show[['Hypertension','Diabetes','Alcoholism','Handicap','NoShow']]
disease_data.head(10)
disease_data.info()
plt.figure(figsize=(15,10))

plt.subplot(2,2,1)

sns.countplot(disease_data.Hypertension,hue=disease_data.NoShow)

plt.subplot(2,2,2)

sns.countplot(disease_data.Diabetes,hue=disease_data.NoShow)

plt.subplot(2,2,3)

sns.countplot(disease_data.Alcoholism,hue=disease_data.NoShow)

plt.subplot(2,2,4)

sns.countplot(disease_data.Handicap,hue=disease_data.NoShow)

plt.show()
