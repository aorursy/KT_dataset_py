#import the liberaries that we need for analysis
import pandas as pd                  #for dealing with dataframs
import numpy as np                   #for scientific compution and arraies
import matplotlib.pyplot as plt      #fro visualization
import seaborn as sns                #fro better visualization
%matplotlib inline
# load the data and first look
df = pd.read_csv("../input/noshowappointments/KaggleV2-May-2016.csv")
df.head()
df.shape
df.info()
df.duplicated().sum()
df.isnull().sum()
df.describe()  #summary statistics
df.nunique()
df.drop(['PatientId', 'AppointmentID'], axis=1, inplace=True)
df.head() # make sure we droped them correctly
df.Age.value_counts()
df = df[df['Age'] >= 0] # drop negative age
df['Age'].value_counts() # make sure there is no negative age
df.info()
df.head()
# change scheduled day and appointment day to datetime 
df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay']).dt.date.astype('datetime64[ns]')
df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay']).dt.date.astype('datetime64[ns]')

df.info() # let's check the datatype again
df['Gender'].value_counts()
df['SMS_received'].value_counts()
df['Scholarship'].value_counts()
df['Neighbourhood'].value_counts()
df.hist(figsize=(10,10));
df['Handcap'] = df[df['Handcap'] >= 1]
df['Handcap'].value_counts()
# Rename incorrect columns names
df = df.rename(columns={'Handcap':'Handicap', 'Hipertension':'Hypertension'})
df.columns
df.head()
df['No-show'].value_counts()
# rename the No-show column to avoid misleading

df = df.rename(columns={'No-show':'Absent'})
df.columns
df['Gender'].value_counts()
sns.countplot(x=df['Absent'], hue=df['Gender']);
plt.title('Male vs Female attendace');
plt.figure(figsize=(20,10))
sns.countplot(x=df.Neighbourhood);
plt.title('Atendance by Neighborhood')
plt.xticks(rotation=90);
plt.figure(figsize=(20, 10))
sns.countplot(x=df['Neighbourhood'], hue=df['Absent']);
plt.xticks(rotation=90);
plt.title('Attendance in different Neighborhood');
df['Age'].hist(bins=10);
df['Age'] = [round(a,-1) for a in df['Age']]  # this trick makes age easier as I divided them into segments to make 
                                                #it easier visualizing
df['Age'].value_counts()
plt.figure(figsize=(20,5))
sns.countplot(x=df['Age'], hue=df['Absent'])
plt.xticks(rotation=90);
df.info()
disease_columns = df[['Hypertension','Diabetes','Alcoholism','Handicap']]
plt.figure(figsize=(20,10));
plt.subplot(2,2,1)
sns.countplot(disease_columns['Hypertension'],hue=df['Absent'])
plt.subplot(2,2,2)
sns.countplot(disease_columns['Diabetes'],hue=df['Absent'])
plt.subplot(2,2,3)
sns.countplot(disease_columns['Alcoholism'],hue=df['Absent'])
plt.subplot(2,2,4)
sns.countplot(disease_columns['Handicap'],hue=df['Absent'])
