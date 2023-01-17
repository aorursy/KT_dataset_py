import numpy as np        
import pandas as pd

#libraries for visualization
import matplotlib.pyplot as plt
import seaborn as sns 
sns.set_style('whitegrid')

#to ignore warning
import warnings
warnings.filterwarnings('ignore')

#to convert dates to week of the day
import datetime  
from datetime import date
import calendar 

#importing the dataset
data = pd.read_csv('../input/Medical.csv')
# To see the complete about the data
data.info()
# to see the first 5 objects of the data.
data.head()
#Rename columns
data.rename(columns = {'Hipertension':'Hypertension',
                      'Handcap':'Handicap',
                      'No-show':'No_show'}, inplace = True)
# Checking for errors and see all the unique values for some features
col = data.columns
for x in col[5:]:
    print(x + ':', sorted(data[x].unique()))
# drop all the values other than 0 and 1
data.drop(data[data['Handicap'].isin([2,3,4])].index,inplace = True)
# drop all the ages less than 0 and more than 100
data = data[(data['Age']<100) & (data['Age']>0)]
dummy = pd.get_dummies(data['No_show'])
dummy.drop('Yes', axis=1, inplace = True)
data.drop('No_show', axis = 1, inplace=True)
data = data.join(dummy)
data.head()
#Rename columns
data.rename(columns = {'No':'Show'}, inplace = True)
# To calculate the day of the appointment
data.AppointmentDay = data.AppointmentDay.apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%dT%H:%M:%SZ').date())
data.ScheduledDay = data.ScheduledDay.apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%dT%H:%M:%SZ').date())

data['DayOfTheWeek'] = data.AppointmentDay.apply(lambda x: calendar.day_name[x.weekday()])
# to calculate the number of days between Appointment Day and Schedule Day
AwaitingDays = (data.AppointmentDay - data.ScheduledDay).astype('timedelta64[D]')
data['AwaitingDays'] = AwaitingDays

sns.stripplot(data = data, y = 'AwaitingDays', jitter = True,color='Green')
plt.ylim(0, 200)
plt.show()
data = data[data.AwaitingDays<=150]
#we check for the duplicate records of patients.
data.PatientId = data.PatientId.apply('int64')

data.PatientId.value_counts().head()
data.drop_duplicates('PatientId', inplace=True)
data.info()
fig,ax=plt.subplots(figsize=(10,7))

sizes = [len(data[data['Show']==0]), len(data[data['Show']==1])]
plt.pie(sizes, autopct='%1.1f%%',colors=['Red','Blue'],labels=['No- show','Show'])
plt.title('Percentage of patients for appointment')
plt.show()
print('The percentage of handicap patients showed up are: '+str(data[(data['Show']==1) & (data['Handicap']==1)].shape[0]/data[(data['Handicap']==1)].shape[0]*100) +'%')
print('The percentage of not handicap patients showed up are: '+str(data[(data['Show']==1) & (data['Handicap']==0)].shape[0]/data[(data['Handicap']==0)].shape[0]*100) +'%')
print('#'*20)
print('The percentage of Hypertension patients showed up are: '+str(data[(data['Show']==1) & (data['Hypertension']==1)].shape[0]/data[(data['Hypertension']==1)].shape[0]*100) +'%')
print('The percentage of no Hypertension patients showed up are: '+str(data[(data['Show']==1) & (data['Hypertension']==0)].shape[0]/data[(data['Hypertension']==0)].shape[0]*100) +'%')
print('#'*20)
print('The percentage of Diabetes patients showed up are: '+str(data[(data['Show']==1) & (data['Diabetes']==1)].shape[0]/data[(data['Diabetes']==1)].shape[0]*100) +'%')
print('The percentage of no Diabetes patients showed up are: '+str(data[(data['Show']==1) & (data['Diabetes']==0)].shape[0]/data[(data['Diabetes']==0)].shape[0]*100) +'%')
print('#'*20)
print('The percentage of Alcoholism patients showed up are: '+str(data[(data['Show']==1) & (data['Alcoholism']==1)].shape[0]/data[(data['Alcoholism']==1)].shape[0]*100) +'%')
print('The percentage of no Alcoholism patients showed up are: '+str(data[(data['Show']==1) & (data['Alcoholism']==0)].shape[0]/data[(data['Alcoholism']==0)].shape[0]*100) +'%')
print('#'*20)
print('The percentage of SMS_received patients showed up are: '+str(data[(data['Show']==1) & (data['SMS_received']==1)].shape[0]/data[(data['SMS_received']==1)].shape[0]*100) +'%')
print('The percentage of not SMS_received patients showed up are: '+str(data[(data['Show']==1) & (data['SMS_received']==0)].shape[0]/data[(data['SMS_received']==0)].shape[0]*100) +'%')

# fuction to find the probability
def probability(group):
    rows=[]
    for item in group:
        for level in data[item].unique():
            row = {'Condition':item}
            total = len(data[data[item] == level])
            n = len(data[(data[item] == level) & (data['Show']== 1)])
            row.update({'Level' : level, 'Probability':n/total})
            rows.append(row)
    return pd.DataFrame(rows)
fig, ax= plt.subplots(figsize=(10,7))
sns.barplot(data = probability(['Scholarship', 'Hypertension','Diabetes','Alcoholism','Handicap']),
           x = 'Condition', y='Probability', hue ='Level', palette ='Set1' )
plt.title('Probability of showing up')
plt.ylabel('Probability')
plt.show()
# pie chart of the no-shows and shows for both genders

fig,ax=plt.subplots(figsize=(10,7))
labels = ['No-Show - Female', 'Show - Female', 'Show - Male', 'No-Show - Male']
sizes = [len(data[(data['Show'] ==0) & (data['Gender'] == 'F')]),len(data[(data['Show'] == 1) & (data['Gender'] == 'F')]),len(data[(data['Show'] == 1) & (data['Gender'] == 'M')]),len(data[(data['Show'] == 0) & (data['Gender'] == 'M')])]
plt.pie(sizes, labels=labels, autopct='%1.1f%%')
plt.show()
#Day of the week

df = data[data['Show']==1]
df = df[['DayOfTheWeek','Show']]
day_group = df.groupby(['DayOfTheWeek'])
day_total = day_group.sum()
day_total.reset_index(inplace=True)
day_total.sort_values('Show',ascending=False,inplace=True)

fig, ax = plt.subplots(figsize=(10,8))
sns.barplot(x = 'DayOfTheWeek', y='Show', data =day_total,palette ='Set1')
plt.title('Number of patients showed up')
plt.ylabel('Number')
plt.show()
# Number of patients for each neighbourhood
df = data[data['Show']==1]
df = df[['Neighbourhood','Show']]
Nei_grp = df.groupby(['Neighbourhood'])
total = Nei_grp.sum()
total.reset_index(inplace=True)
total.sort_values('Show',ascending=False,inplace=True)
fig,ax=plt.subplots(figsize=(20,16))
sns.barplot(x='Neighbourhood', y='Show', data=total, ax=ax )
plt.xticks(rotation=80)
plt.title('Number of patients showed up')
plt.ylabel('Number')
plt.show()
# Patients who received and not received the SMS notification
df = data[data['Show']==1]
df = df[['SMS_received','Show']]
SMS_grp = df.groupby(['SMS_received'])
total = SMS_grp.sum()
total.reset_index(inplace=True)
total.sort_values('Show',ascending=False,inplace=True)
fig,ax=plt.subplots(figsize=(10,7))
sns.barplot(x='SMS_received', y='Show', data=total, ax=ax )
plt.title('Number of patients showed up')
plt.ylabel('Number')
plt.show()
# Distribution of Age
plt.figure(figsize=(10,8))
sns.distplot(data['Age'], bins=10, kde=False)
plt.xlabel("Age")
plt.ylabel("Patients")
plt.title('Distribution of Age')
plt.show()
#Distribution of Age with respect to Gender
fig, ax = plt.subplots(figsize=(10,8))
df = data[data.Show == 1]
range_df = pd.DataFrame()
range_df['Age'] = range(100)
men = range_df.Age.apply(lambda x: len(df[(df.Age == x) & (df.Gender == 'M')]))
women = range_df.Age.apply(lambda x: len(df[(df.Age == x) & (df.Gender == 'F')]))
plt.plot(range(100),men, 'b')
plt.plot(range(100),women, color = 'r')
plt.legend(['M','F'])
plt.xlabel('Age')
plt.title('Women visit the doctor more often')
plt.show()
#Age wise distribution of diseases
fig, ax = plt.subplots(nrows =2, ncols=2, figsize=(15,10))
df_hyper = data[data['Hypertension']==1]
df_hyper = df_hyper[['Hypertension','Age']]

df_diab = data[data['Diabetes']==1]
df_diab = df_diab[['Diabetes','Age']]

df_alco = data[data['Alcoholism']==1]
df_alco = df_alco[['Alcoholism','Age']]

df_handi= data[data['Handicap']==1]
df_handi = df_handi[['Handicap','Age']]


sns.distplot(df_hyper['Age'], bins=10, kde=False,ax=ax[0,0])
ax[0,0].set_xlabel("Age")
ax[0,0].set_ylabel("Patients of Hypertension")
ax[0,0].set_title('Distribution of Age with Hypertension')

sns.distplot(df_diab['Age'], bins=10, kde=False,ax=ax[0,1])
ax[0,1].set_xlabel("Age")
ax[0,1].set_ylabel("Patients of Diabetes")
ax[0,1].set_title('Distribution of Age with Diabetes')

sns.distplot(df_alco['Age'], bins=10, kde=False,ax=ax[1,0])
ax[1,0].set_xlabel("Age")
ax[1,0].set_ylabel("Patients of Alcoholism ")
ax[1,0].set_title('Distribution of Age with Alcoholism')

sns.distplot(df_handi['Age'], bins=10, kde=False,ax=ax[1,1],)
ax[1,1].set_xlabel("Age")
ax[1,1].set_ylabel("Patients of Handicap")
ax[1,1].set_title('Distribution of Age with Handicap')
plt.show()
