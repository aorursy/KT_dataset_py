import pandas as pd

import numpy as np

import matplotlib.pyplot as plt 

import seaborn as sns

%matplotlib inline
df= pd.read_csv("../input/noshowappointments/KaggleV2-May-2016.csv")
df.head()
df.info()
df.shape
df.describe()
df['Age'].value_counts() 
df = df[(df.Age >= 0)]
df[(df.Age <= 0) & (df.Alcoholism == 1)]
df['Gender'].value_counts() 
df['Scholarship'].value_counts() 
df['Hipertension'].value_counts() 
df['Diabetes'].value_counts() 
df['Alcoholism'].value_counts() 
df['Handcap'].value_counts() 
df['SMS_received'].value_counts() 
df['No-show'].value_counts() 
df.hist(figsize=(10,8));
df.isnull().sum().any()

df.duplicated().sum()
df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay']).dt.date.astype('datetime64[ns]')

df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay']).dt.date.astype('datetime64[ns]')
df.drop(['PatientId','AppointmentID'],axis=1 , inplace=True)
df.head()
df['Gender'].value_counts() 

df['No-show'].value_counts() 
ax = sns.countplot(x='No-show', data=df)
ax = sns.countplot(x="No-show", hue="Gender", data=df)

g.figzie=(30,10)

g = sns.countplot(x="Age", hue="No-show" ,data=df)
df['Neighbourhood'].value_counts().plot(kind='barh',figsize=(30,15))
ax.figzie=(30,10)

ax = sns.countplot(x="Neighbourhood", hue="No-show" ,data=df)
ax = sns.countplot(x='Scholarship', hue='No-show', data=df)

ax.set_title("Show/NoShow for Scholarship")

x_ticks_labels=['No Scholarship', 'Scholarship']

ax.set_xticklabels(x_ticks_labels)

plt.show()
ax = sns.countplot(x='Hipertension', hue='No-show', data=df)

ax.set_title("Show/NoShow for Hipertension")

x_ticks_labels=['No Hipertension', 'Hipertension']

ax.set_xticklabels(x_ticks_labels)

plt.show()
ax = sns.countplot(x='Diabetes', hue='No-show', data=df)

ax.set_title("Show/NoShow for Diabetes")

x_ticks_labels=['No Diabetes', 'Diabetes']

ax.set_xticklabels(x_ticks_labels)

plt.show()
ax = sns.countplot(x='Alcoholism', hue='No-show', data=df)

ax.set_title("Show/NoShow for Alcoholism")

x_ticks_labels=['No Alcoholism', 'Alcoholism']

ax.set_xticklabels(x_ticks_labels)

plt.show()
ax = sns.countplot(x='Handcap', hue='No-show', data=df)

ax.set_title("Show/NoShow for Handcap")

x_ticks_labels=['No Handcap', 'Handcap']

ax.set_xticklabels(x_ticks_labels)

plt.show()
df['Handcap'].value_counts().sum()
df['Handcap'].value_counts()
cat = df[["Handcap", "No-show"]]
cat = cat[(cat['No-show'] == 'No')]

cat['Handcap'].value_counts()
ax = sns.countplot(x='SMS_received', hue='No-show', data=df)

ax.set_title("Show/NoShow for SMS received")

x_ticks_labels=['No SMS','SMS']

ax.set_xticklabels(x_ticks_labels)

plt.show()