#All import needed
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
#Perform operations to inspect data
df = pd.read_csv('/kaggle/input/noshowappointments/KaggleV2-May-2016.csv')
df.head()
df.info()
df.describe()
# This report like describe-function but in a report that contains more details about all the variables with the interactions 
# and some of the warnings about the dataset.
from pandas_profiling import ProfileReport
profile = ProfileReport(df)
profile.to_widgets()
#PatientId and AppointmentID did not need in the analysis it too related to the patient information, drop them
df.drop(['PatientId','AppointmentID'], axis=1, inplace=True)
df.info()
#Change type for ScheduledDay to datetime
df['ScheduledDay']=pd.to_datetime(df['ScheduledDay'])
df['ScheduledDay']
#Change type for AppointmentDay to datetime
df['AppointmentDay']=pd.to_datetime(df['AppointmentDay'])
df['AppointmentDay']
#Based on the report we see the minimum value in age is -1 which is incorrect 
check_valid_Age = df[df['Age']<0].index
check_valid_Age
#Drop Age = -1 by the index
df.drop(check_valid_Age , inplace=True)
df['Age'].describe()
#Based on the report we see there are some values greater than 1 in Handicap it maybe considers as categories situation or something more specific not known 
check_valid_Handcap = df[df['Handcap']>1].index
check_valid_Handcap 
#Drop categorizes handicap since it will not affect the analysis
df.drop(check_valid_Handcap  , inplace=True)
df['Handcap'].value_counts()
#Change datatype of variables 
df['Scholarship']=df['Scholarship'].astype('bool')
df['Hipertension']=df['Hipertension'].astype('bool')
df['Diabetes']=df['Diabetes'].astype('bool')
df['Alcoholism']=df['Alcoholism'].astype('bool')
df['Handcap']=df['Handcap'].astype('bool')
df['SMS_received']=df['SMS_received'].astype('bool')
#Rename the No-Show to be readable
df.rename(columns={'No-show':'Attende'}, inplace=True)
#To chaeck after change after Cleaning
df.info()
#Save data after cleaning
df.to_csv('No-show-clean.csv', index=False)
df_clean = pd.read_csv('No-show-clean.csv')
df_clean.head()
df_clean['Age'].hist(color='lightblue');
df_GenderAttende= df_clean.groupby('Gender', as_index=False).Attende.count()
df_GenderAttende
#Who is attende most of the appointment
plt.bar(df_GenderAttende['Gender'],df_GenderAttende['Attende'],color=['lightgray','lightblue'])
plt.title('Male and Female attende')
plt.xlabel('Gender')
plt.ylabel('Attende Appointments');
df_AttendeSMS= df_clean.groupby('Attende', as_index=False).SMS_received.count()
df_AttendeSMS
#Is attende effect by sending SMS?
plt.bar(df_AttendeSMS['Attende'],df_AttendeSMS['SMS_received'],color=['lightgray','lightblue'])
plt.title('SMS Affect with attencene')
plt.xlabel('Attende Appointments')
plt.ylabel('number of recive sms');
df_GenderScholarship = df_clean.groupby('Gender', as_index=False).Scholarship.count()
df_GenderScholarship
#Is scolarship in male or female effect attend?
plt.bar(df_GenderScholarship['Gender'],df_GenderScholarship['Scholarship'],color=['lightgray','lightblue'])
plt.title('Gender Scholarship')
plt.xlabel('Gender')
plt.ylabel('Number of who is educate');
#in which month have most unateend ...
df_clean['AppointmentDay']=pd.to_datetime(df_clean['AppointmentDay'])
df_clean['AppointmentDay']
df_clean['AppointmentMonth']=df_clean.AppointmentDay.dt.month_name()
df_clean['AppointmentMonth']
df_Months = df_clean.groupby('AppointmentMonth')['Attende'].count()
df_Months
#In which month did the patients show up to their appointments
df_Months.plot(kind='bar',color='lightblue');
df_Alcoholism= df_clean.groupby('Attende', as_index=False).Alcoholism.count()
df_Alcoholism
#many patient has alcoholic didn't come to there appointments
plt.bar(df_Alcoholism['Attende'],df_Alcoholism['Alcoholism'],color=['lightgray','lightblue'])
plt.title('Male and Female attende who is Alcoholism')
plt.xlabel('Attende Appointments')
plt.ylabel('rate of Alcoholism');