# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# read data

df = pd.read_csv('../input/No-show-Issue-Comma-300k.csv')



# A bit of preprocessing

df['Age'] = df['Age'].clip(lower=0, upper=None)



df['Status'] = df['Status'].astype('category')

df = df.join(pd.get_dummies(df['Status']))



df['AppointmentRegistration'] = pd.to_datetime(df['AppointmentRegistration'])

df['AppointmentData'] = pd.to_datetime(df['ApointmentData'])

df = df.drop('ApointmentData', 1)



# add dayofweek as a number and as separate one-hot encodings

df['dayofweek'] = df['AppointmentData'].dt.dayofweek

df = df.join(pd.get_dummies(df['DayOfTheWeek']))
df.describe()
df2 = df.groupby('Age').sum()

df2['ShowUpRatio'] = df2['Show-Up'] / (df2['Show-Up'] + df2['No-Show'])

df2[['Show-Up', 'No-Show']].plot()

df2[['ShowUpRatio']].plot()
df2 = df.groupby('dayofweek')['Show-Up','No-Show'].sum()

df2['ShowUpRate'] = df2['Show-Up'] / (df2['Show-Up'] + df2['No-Show'])

df2[['Show-Up','No-Show']].plot(kind='bar',stacked=True)

df2
df2 = df.groupby('Sms_Reminder')[['Show-Up','No-Show']].sum()

df2['ShowRatio'] = df2['Show-Up'] / (df2['Show-Up'] + df2['No-Show'])

df2
df2 = df.groupby('Gender')[['Show-Up','No-Show']].sum()

df2['ShowRatio'] = df2['Show-Up'] / (df2['Show-Up'] + df2['No-Show'])

df2
df_await = df[df['AwaitingTime'] > -50]

df2 = df_await.groupby('AwaitingTime').sum()

df2['ShowRatio'] = df2['Show-Up'] / (df2['Show-Up'] + df2['No-Show'])

df2[['Show-Up', 'No-Show']].plot()

df2[['ShowRatio']].plot()
# Diabetes Alcoolism HiperTension Handicap Scolarship Tuberculosis

import IPython



for col in 'Diabetes,Smokes,Alcoolism,HiperTension,Handcap,Scholarship,Tuberculosis'.split(','):

    df2 = df.groupby(col)[['Show-Up','No-Show']].sum()

    df2['ShowRatio'] = df2['Show-Up'] / (df2['Show-Up'] + df2['No-Show'])

    print (df2)

    
# I think time of day of the appointment could be a valuable source of information,

# but I suppose it's omitted for privacy reasons.



# Just for fun, let's do the exercise using the appointment registration time.

reg = df['AppointmentRegistration']



# strip the date (+truncate a bit so we have a reasonable amount of buckets)

## 2nd graph is useless if we don't drop the second accuracy; maybe blocks of 5-15 minutes would be even better

##df['AppointmentRegistrationTimeOfDay'] = reg - reg.dt.floor('D')

df['AppointmentRegistrationTimeOfDay'] = reg.dt.floor('MIN') - reg.dt.floor('D')



df2 = df.groupby('AppointmentRegistrationTimeOfDay').sum()

df2['ShowRatio'] = df2['Show-Up'] / (df2['Show-Up'] + df2['No-Show'])

df2[['Show-Up', 'No-Show']].plot()

df2[['ShowRatio']].plot()
# Final note: Ignoring the date/time related columns, there is a lot of duplicate information