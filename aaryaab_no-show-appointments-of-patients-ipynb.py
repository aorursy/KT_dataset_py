import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import statsmodels.api as sm

df=pd.read_csv('../input/no-show-patients/NS2016.csv')
df.head()
df.info()
df1=df.copy()
df1=df1.drop(['PatientId','ScheduledDay','AppointmentDay','Hipertension','Diabetes','Alcoholism','Handcap'],axis=1)
df1=df1.rename({'No-show':'Show-Up'},axis=1)
df1['Show-Up']=df1['Show-Up'].replace('No',1)
df1['Show-Up']=df1['Show-Up'].replace('Yes',0)
df1.head()
df1.count()
round(((df1['Show-Up'].value_counts()/df1['Show-Up'].count())*100),2)
df1['Show-Up'].value_counts().plot(kind='pie',autopct='%1.0f%%',figsize=(15,6),labels=None)
plt.legend(labels=df1['Show-Up'].unique(), loc='lower left') 
plt.show()
round(((df1['Gender'].value_counts()/df1['Gender'].count())*100),2)
b=df1.groupby(['Gender','Show-Up']).size()
round(((b/df1['Gender'].count())*100),2)
df1.groupby(['Gender','Show-Up']).size().plot(kind='pie',labels=None,autopct='%1.0f%%',figsize=(18,6),pctdistance=1.12)
plt.legend(labels=['Female No Show-Up','Female Show-Up','Male No Show-Up','Male Show-Up'],loc='lower right',bbox_to_anchor=(0.75,0.75),
          bbox_transform=plt.gcf().transFigure)
plt.show()
df1.groupby(['Gender','Show-Up'])['Show-Up'].count().unstack('Show-Up').plot(kind='bar',stacked=False,figsize=(12,6))
plt.title('Gender variation in showing up at appointment')
age_g=[0,5,10,20,30,40,50,60,70,80,90]
age=['0','5','10','20','30','40','50','60','70','80']
# O means below 5 years of age, which means an infant child
df1['age_group']=pd.cut(df['Age'],age_g,labels=age)
df1.head()
plt.figure(figsize=(15,2))
sb.boxplot(x=df1.Age)
plt.title('Age distribution of Patients')
plt.figure(figsize=(14,6))
sb.countplot(x=df1.age_group)
plt.title("Appointments made by different age groups")
plt.figure(figsize=(14,6))
sb.countplot(x=df1.Age)
plt.title("No of patients by age ")
df1.groupby(['age_group', 'Show-Up'])['age_group'].count().unstack('Show-Up').plot( kind='bar', stacked=False)

round(((df1.groupby(['Scholarship','Show-Up']).size()/df1['Gender'].count())*100),2)
df1.groupby(['Scholarship','Show-Up'])['Show-Up'].count().unstack('Show-Up').plot(kind='bar',stacked=False)
round(((df1.groupby(['SMS_received','Show-Up']).size()/df1['Gender'].count())*100),2)
df1.groupby(['SMS_received','Show-Up'])['Show-Up'].count().unstack('Show-Up').plot(kind='bar',stacked=False)
plt.title("Impact of SMS sent on show ups at appointment")
df2=df.copy()
df2=df2.drop(['PatientId'],axis=1)
df2=df2.rename({'No-show':'Show-Up'},axis=1)
df2['Show-Up']=df2['Show-Up'].replace('No',1)
df2['Show-Up']=df2['Show-Up'].replace('Yes',0)
df2.head()
df2['ScheduledDay']=pd.to_datetime(df['ScheduledDay']).dt.date.astype('datetime64[ns]')
df2['AppointmentDay']=pd.to_datetime(df['AppointmentDay']).dt.date.astype('datetime64[ns]')
df2['awaiting_time_days'] = (df2.AppointmentDay - df2.ScheduledDay).dt.days
df2.info()
df2['AppointmentDay']=df2.ScheduledDay.dt.day_name()
df2['AppointmentDay'].value_counts()
df2.groupby(['AppointmentDay','Show-Up'])['Show-Up'].count().unstack('Show-Up').plot(kind='bar',stacked=False,figsize=(15,6))
plt.title('Relation of weekdays with appointment show ups')
c=df2['awaiting_time_days'].unique()
c.sort()
c
disease = ['Hipertension', 'Diabetes', 'Alcoholism', 'Handcap']

fig = plt.figure(figsize=(14, 10))
for i, var in enumerate(disease):
    ax = fig.add_subplot(2, 2, i+1)
    df2.groupby([var, 'Show-Up'])[var].count().unstack('Show-Up').plot(ax=ax, kind='bar', stacked=False)
