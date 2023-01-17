import pandas as pd 

import numpy as np 

import matplotlib.pyplot as plt 

import seaborn as sns

import datetime as dt



%matplotlib inline



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/udacitybrazilmedicalappointments/noshowappointments-kagglev2-may-2016.csv')

df.head()
df.shape
df.describe()
df.isnull().sum()
df.columns
#Renaming the misspelt columns 

df = df.rename(columns = {"Handcap" : "Handicap" , 'Hipertension' : 'Hypertension' , 'SMS_received':'SMS_recieved'})
df.dtypes
df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'], utc= None ).dt.normalize()

df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'], utc= None).dt.normalize()

df.info()
for col in ['Scholarship' , 'Diabetes', 'Alcoholism' , 'Handicap', 'SMS_recieved']:

    df[col] = df[col].astype('bool')
df.dtypes
df.isnull().sum()
#remove invalid records

df = df.query('Age >=0')

df.info()
for ech in df.columns :

    print('Column: ',ech ,':',df[ech].duplicated().sum())
df['PatientId'].duplicated().sum() /df['AppointmentID'].count()
df.info()
df.query('Gender == "F" ').count()/df['AppointmentID'].count()
df.query('Gender == "M" ').count()/df['AppointmentID'].count()
df.groupby('Gender')['Gender'].count().plot(kind= 'bar')
plt.figure(figsize=(10,10))

sns.distplot(df['Age'])
sns.boxplot(df['Age'])
df.groupby(['SMS_recieved' , 'No-show'])['No-show'].count().plot(kind = 'bar')
df
df['SchedulDay'] = df['AppointmentDay'].dt.day_name()
df.groupby('SchedulDay').count()['AppointmentID'].plot(kind = 'bar')
df['DaysBeforeAppointment'] = (df['AppointmentDay'] - df['ScheduledDay']).dt.days
df.info()
df['DaysBeforeAppointment'].max()

df['AppointmentGroups'] = pd.cut(x = df['DaysBeforeAppointment'] , bins = [0,2, 179] , labels =['urgent','normal'] , right = False )

df
df.groupby('AppointmentGroups').count()['AppointmentID'].plot(kind = 'bar')
sns.countplot(df['AppointmentGroups'] , hue = df['No-show'])
sns.countplot(df['SchedulDay'], hue = df['No-show'] )
df['AgeGroups'] =pd.cut(x = df['Age'] , bins =[0,15, 25 , 50 , 70 , 115], labels = [ '0<15','16 < 25', '26 < 50' , '51 < 70' , '70 < 115'] , right = True)
countmatrix = df.query('Hypertension == True').groupby(['Hypertension' , 'AgeGroups'])['AppointmentID'].count()
d = df.query('Hypertension == True')['AgeGroups'].value_counts()
plt.bar(d.index , d.values)
countmatrix = df.query('Hypertension == True').groupby(['AgeGroups'])['AppointmentID'].count()
countmatrix
plt.bar(countmatrix.index , countmatrix.values)
scholarshipmatrix = df.groupby(['Scholarship', 'No-show' ])['AppointmentID'].count()

scholarshipmatrix
scholarshipmatrix = scholarshipmatrix.astype(int)
#plt.bar(scholarshipmatrix.index , scholarshipmatrix.values)

scholarshipmatrix.plot(kind='bar')
fig , ax = plt.subplots()
patients = df.groupby(['Neighbourhood', 'AgeGroups'])['AppointmentID'].count()
patient_data = pd.DataFrame(patients)
patient_data.reset_index(inplace=True)

patient_data
data = patient_data.pivot(columns='AgeGroups',index = 'Neighbourhood',  values = 'AppointmentID')

data =data.fillna(0)

data = data.astype(int)

data
data.values
plt.figure(figsize=(20,20))



sns.set(font_scale=1) # font size 2



sns.heatmap(data, annot = True)
