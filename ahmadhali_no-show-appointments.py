import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
import calendar

%matplotlib inline
df = pd.read_csv('../input/KaggleV2-May-2016.csv')
df.head()
df.info()
df.rename(columns=lambda x: x.lower(), inplace=True)
df.rename(columns={'patientid':'patient_id', 'appointmentid':'appointment_id', 
                   'scheduledday':'scheduled_day', 'appointmentday':'appointment_day',
                   'neighbourhood':'neighborhood', 'scholarship':'bolsa_familia',
                   'hipertension':'hypertension', 'handcap':'handicap',
                   'no-show':'no_show'}, inplace=True)
print('Gender:',df.gender.unique())
print('Age:',sorted(df.age.unique()))
print('Neighborhood:',df.neighborhood.unique())
print('Bolsa Familia:',df.bolsa_familia.unique())
print('Hypertension:',df.hypertension.unique())
print('Diabetes:',df.diabetes.unique())
print('Alcoholism:',df.alcoholism.unique())
print('Handicap:',df.handicap.unique())
print('SMS Received:',df.sms_received.unique())
print('No-show:',df.no_show.unique())
df = df[(df.age >= 0) & (df.age <= 100)]
df['scheduled_day'] = pd.to_datetime(df['scheduled_day'])
df['appointment_day'] = pd.to_datetime(df['appointment_day'])
df.no_show.replace(to_replace=dict(Yes='no_show', No='show'), inplace=True)
df['waiting_days'] = df['appointment_day'] - df['scheduled_day']
df['waiting_days'] = df['waiting_days'].astype('timedelta64[D]')
df['waiting_days'] = np.where(df['waiting_days'] < 0, 0, df['waiting_days'])
df['appointment_weekday'] = df['appointment_day'].dt.weekday_name
df['scheduled_weekday'] = df['scheduled_day'].dt.weekday_name
df.head()
mean_no_show = df['no_show'].value_counts()[1] / len(df['no_show'])

mean_no_show

def groupby_rate(column):
    #Return a series with the no-show rates for a specific characteristic (column).
    
    func_count = df.groupby(['no_show', column]).count()['patient_id']
    func_rate = func_count['no_show'] / (func_count['no_show'] + func_count['show'])
    return func_rate
sort = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']

apt_weekdays_rate = groupby_rate('appointment_weekday').reindex(sort, copy=False);
ind = np.arange(len(apt_weekdays_rate))
width = 0.35
plt.bar(ind, apt_weekdays_rate, width, color='r', alpha=.7, label='No-Show')
location = ind
plt.ylabel('No-show Rate')
plt.xlabel('Weekday')
plt.title('No-show Rate by Appointment Weekday')
labels = sort
plt.xticks(location, labels)
plt.yticks(np.arange(0, 0.4, step=0.05))
plt.legend();
print(apt_weekdays_rate)
sch_weekdays_rate = groupby_rate('scheduled_weekday').reindex(sort, copy=False);
ind = np.arange(len(sch_weekdays_rate))
width = 0.35
plt.bar(ind, sch_weekdays_rate, width, color='r', alpha=.7, label='No-Show')
location = ind
plt.ylabel('No-show Rate')
plt.xlabel('Weekday')
plt.title('No-show Rate by Scheduled Weekday')
labels = sort
plt.xticks(location, labels)
plt.yticks(np.arange(0, 0.4, step=0.05))
plt.legend();
print(sch_weekdays_rate)
gender_rate = groupby_rate('gender')
bolsa_rate = groupby_rate('bolsa_familia')
hypertension_rate = groupby_rate('hypertension')
diabetes_rate = groupby_rate('diabetes')
alcoholism_rate = groupby_rate('alcoholism')
handicap_rate = groupby_rate('handicap')
ind = np.arange(len(gender_rate))
width = 0.35
plt.bar(ind, gender_rate, width, color='r', alpha=.7, label='No-Show')
location = ind
plt.ylabel('No-show Rate')
plt.xlabel('Gender')
plt.title('No-show Rate by Gender')
plt.xticks(location)
plt.yticks(np.arange(0, 0.4, step=0.05))
plt.legend();
print(gender_rate)
ind = np.arange(len(bolsa_rate))
width = 0.35
plt.bar(ind, bolsa_rate, width, color='r', alpha=.7, label='No-Show')
location = ind
plt.ylabel('No-show Rate')
plt.xlabel('Bolsa-Familia')
plt.title('No-show Rate by Bolsa-Familia recievers')
plt.xticks(location)
plt.yticks(np.arange(0, 0.4, step=0.05))
plt.legend();
print(bolsa_rate)
ind = np.arange(len(hypertension_rate))
width = 0.35
plt.bar(ind, hypertension_rate, width, color='r', alpha=.7, label='No-Show')
location = ind
plt.ylabel('No-show Rate')
plt.xlabel('Hypertension')
plt.title('No-show Rate by Hypertension')
plt.xticks(location)
plt.yticks(np.arange(0, 0.4, step=0.05))
plt.legend();
print(hypertension_rate)
ind = np.arange(len(diabetes_rate))
width = 0.35
plt.bar(ind, diabetes_rate, width, color='r', alpha=.7, label='No-Show')
location = ind
plt.ylabel('No-show Rate')
plt.xlabel('Diabetes')
plt.title('No-show Rate by Diabetes')
plt.xticks(location)
plt.yticks(np.arange(0, 0.4, step=0.05))
plt.legend();
print(diabetes_rate)
ind = np.arange(len(alcoholism_rate))
width = 0.35
plt.bar(ind, alcoholism_rate, width, color='r', alpha=.7, label='No-Show')
location = ind
plt.ylabel('No-show Rate')
plt.xlabel('Alcoholism')
plt.title('No-show Rate by Alcoholism')
plt.xticks(location)
plt.yticks(np.arange(0, 0.4, step=0.05))
plt.legend();
print(alcoholism_rate)
ind = np.arange(len(handicap_rate))
width = 0.35
plt.bar(ind, handicap_rate, width, color='r', alpha=.7, label='No-Show')
location = ind
plt.ylabel('No-show Rate')
plt.xlabel('Handicap')
plt.title('No-show Rate by Handicaps')
plt.xticks(location)
plt.yticks(np.arange(0, 0.4, step=0.05))
plt.legend()
print(handicap_rate)
bins = [0, 18, 34, 50, 70, 100]

age_group = df.groupby(['no_show', pd.cut(df['age'], bins)]).size().unstack().transpose()

age_group_rate = age_group['no_show'] / (age_group['no_show'] + age_group['show'])
ind = np.arange(len(age_group_rate))
width = 0.35
plt.bar(ind, age_group_rate, width, color='r', alpha=.7, label='No-Show')
location = ind
plt.ylabel('No-show Rate')
plt.xlabel('Age Group')
plt.title('No-show Rate by Age Group')
labels = ['0-18', '19-34', '35-50', '51-70', '71-100']
plt.xticks(location, labels)
plt.yticks(np.arange(0, 0.4, step=0.05))
plt.legend();
print(age_group_rate)
neighborhood_rate = groupby_rate('neighborhood')
neighborhood_count = df['neighborhood'].value_counts()
neighborhood_cmb = pd.concat([neighborhood_rate, neighborhood_count], axis=1, sort=False)

neighborhood_cmb.rename(columns={'patient_id':'no_show_rate', 'neighborhood':'size'}, inplace=True)
plt.scatter(neighborhood_cmb['size'], neighborhood_cmb['no_show_rate'], c='r', alpha=0.7)
plt.ylabel('No-show Rate')
plt.xlabel('Neighborhood Size')
plt.title('No-show Rate by Neighborhood Size');
