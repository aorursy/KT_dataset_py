import os

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

%matplotlib inline



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv('/kaggle/input/noshowappointments/KaggleV2-May-2016.csv')

df.head()
df.describe()
df.info()
# Dropping PatientId and AppointmentID because they are not needed

df.drop(['PatientId','AppointmentID'], axis=1, inplace=True)



# Converting all column names to lower case for convenience

df.columns = map(str.lower, df.columns)



# Rename columns to make calling variables easier

df.rename(columns={'no-show':'no_show', 'appointmentday':'appointment_day', 'scheduledday':'scheduled_day'}, inplace=True)



# Converting the no-shows into binary numbers and data type integer64

df['no_show'].mask(df['no_show'] == 'No', 0, inplace=True)

df['no_show'].mask(df['no_show'] == 'Yes', 1, inplace=True)

df.no_show = df.no_show.astype(int)
# Checking for the patient whose age is below or equal zero

df[df['age'] <= 0]
# We're simply going to fill in all people whose age is equal or below zero with the mean

df.loc[df['age'] <= 0, 'age'] = df['age'].mean()

df['age'].describe()
# First we need to remove unnecessary characters from the date and time stamp

df['appointment_day'] = df.appointment_day.apply(lambda x: x.replace('T00:00:00Z', ''))

df['scheduled_day'] = df.scheduled_day.apply(lambda x: x.replace('T',' '))

df['scheduled_day'] = df.scheduled_day.apply(lambda x: x.replace('Z',''))



# Once that is done, we can easily convert the value of both columns into datetime type

df['appointment_day'] = pd.to_datetime(df['appointment_day'])

df['scheduled_day'] = pd.to_datetime(df['scheduled_day'])



# Confirming that the data type is now correct for all columns

df.info()
df.hist(figsize=(15,15));
# Defining two masks for people who showed up and people who didn't

no_show = df['no_show'] == 1

show = df['no_show'] == 0



# How many people didn't show up?

df.age[no_show].count()/df.age.count()
df[show].age.mean()-df[no_show].age.mean()
df.age[no_show].hist(label='No show', alpha=0.5, bins=10)

df.age[show].hist(label='Show', alpha=0.5, bins=10)

plt.ylabel('Frequency')

plt.xlabel('Age')

plt.title('Distribution of Age')

plt.legend();
df.groupby('no_show').mean().sms_received.plot(kind='bar')

plt.ylabel('Received SMS mean')

plt.xlabel('No Show or Show')

plt.title('Paitents No Show (SMS vs. No SMS)')

ax = plt.gca()

ax.set_xticks([0,1])

ax.set_xticklabels(['Show', 'No Show']);
df.groupby('no_show').mean().sms_received
df[no_show].groupby('neighbourhood').no_show.count().plot(kind='bar', figsize=(25,10));

plt.ylabel('No Shows')

plt.xlabel('Neighbourhood')

plt.title('No Shows by Neighbourhood');
df[no_show].groupby('neighbourhood').no_show.count().max()
df[df['neighbourhood'] == 'JARDIM DA PENHA'].age.mean() - df.age.mean()
df[df['neighbourhood'] == 'JARDIM DA PENHA'].groupby('no_show').handcap.count()
df[df['handcap'] != 0].no_show.mean(), df.no_show.mean()