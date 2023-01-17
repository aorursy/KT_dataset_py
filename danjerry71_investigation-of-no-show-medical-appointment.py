# Import packages

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set_style('darkgrid')
# data is loaded and the first five rows are inspected with the head function

data = pd.read_csv('../input/noshowappointments/KaggleV2-May-2016.csv')

data.head()
# Information about the dataset using info function
data.info()
# Check for duplicates
data.duplicated().value_counts()
# Description of the data
data.describe()
# Number of patients with age -1
data.query('Age == -1').Age.count()
# Check the unique values of Handcap
data.Handcap.unique()
# Check the Handcap values and the number of patients for each.
h = data.Handcap.value_counts()
h
# Percentage of total unwanted Handcap values of 2, 3 and 4.
total = h[2] + h[3] + h[4]
(total/data.Handcap.count())*100
# The columns PatientId, ScheduledDay, AppointmentID, Hipertension, Handcap and No-show are renamed
data.rename(columns={'PatientId':'patient_id','AppointmentID':'appointment_id','ScheduledDay':'scheduled_day','AppointmentDay':'appointment_day','Hipertension':'hypertension','Handcap':'handicap','No-show':'no_show'}, inplace=True)
# Checking the correction
data.head(2)
data.columns = data.columns.map(str.lower)
#check
data.head(2)
# Change the data type to datetime.
data['scheduled_day'] = pd.to_datetime(data['scheduled_day'])
data['appointment_day'] = pd.to_datetime(data['appointment_day'])
# The data values of column No_show is changed: No = 1 and Yes = 0
data['no_show'] = data['no_show'].apply(lambda x: 1 if x == 'No' else 0)

# Check
data.head()
# The index of the row with Age = -1
# Note: KeyError will show because if this cell is rerun more than once because the index has been remove already

age_n = data.query('age ==-1').index.values
data.drop(age_n, inplace=True)
# Remove multiple rows where handicap values are 2, 3 and 4

# create a variable
hdt = data.query('handicap == 2 | handicap == 3 | handicap == 4').handicap

# Drop the values from the dataframe
data.drop(hdt.index, inplace=True)

# check
data.handicap.unique()
# Total number of patient each day that kept up with the appointment
keep_up = data.query('no_show == 1').appointment_day
day_up = keep_up.dt.dayofweek.value_counts()
day_up
# total number that showed up
day_up.sum()
# Calculation of percentages for each day 
Mon = (day_up[0] / day_up.sum())*100
Tue = (day_up[1] / day_up.sum())*100
Wed = (day_up[2] / day_up.sum())*100
Thur = (day_up[3] / day_up.sum())*100
Fri = (day_up[4] / day_up.sum())*100
Sat = (day_up[5] / day_up.sum())*100

# Total percent should be 100%
tot = Mon + Tue + Wed + Thur + Fri + Sat
# Create a dataframe to display the information on a table

d_total = [day_up[0], day_up[1], day_up[2], day_up[3], day_up[4], day_up[5], day_up.sum()]

d_per = [Mon.round(2), Tue.round(2), Wed.round(2), Thur.round(2), Fri.round(2), Sat.round(2), tot.round()]

val ={'Total for each day': d_total, 'Percentage of each day': d_per}

index = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday','Total']

column = ["Total for each day", "Percentage of each day"]

tab = pd.DataFrame(val, index=index, columns=column)

tab
# Query the dataframe for show up
t = data.query('no_show == 1')
a_time = t.appointment_day #appointment_day
s_time = t.scheduled_day # scheduled day

# Difference between the appointment days and scheduled days for each of those showed up using dt.days function. 
wait_time = (a_time - s_time).dt.days
# Waiting time for those that showed up
abs_wait_time = wait_time.abs()

# Average waiting time
mean_wait_time = abs_wait_time.mean()
mean_wait_time
# Describing the patients that showed up
abs_wait_time.describe()
# Query the dataframe for no show
nt = data.query('no_show == 0') # Patients that did not show up
na_time = nt.appointment_day   # Patients that did not show up appointment day
ns_time = nt.scheduled_day     #Patients that did not show up scheduled day

# Difference between the appointment days and scheduled days for patients that did not show up using dt.days function.
nwait_time = (na_time - ns_time).dt.days
# Waiting time for patients that did not show up

abs_nwait_time = nwait_time.abs()

# average waiting time
mean_nwait_time = abs_nwait_time.mean()
mean_nwait_time
# describing the patients that did not show up
abs_nwait_time.describe()
## The array 't' is queried to get the male and female gender

# Male
male_time = t.query('gender=="M"')
male_time.head()
# Difference between the appointment day and scheduled day
male_wait_time = (male_time['appointment_day'] - male_time['scheduled_day']).dt.days.abs()
male_wait_time.describe()
# Females
female_time = t.query('gender == "F"')
female_time.head()
# Difference between the appointment day and scheduled day
female_wait_time = (female_time['appointment_day'] - female_time['scheduled_day']).dt.days.abs()
female_wait_time.describe()
# Intervals of the ages
age_interval = pd.cut(t['age'], bins=[0, 15, 25, 55, 65, 116], labels=['Children <0-14>','Early<15-24>','Prime<25-54>','Mature<55-64>','Elderly<65+>'],include_lowest=True)
age_interval.value_counts()
# shorter name
col = age_interval.value_counts()
# The values in percent
children = ((col["Children <0-14>"]/col.sum())*100).round(2)
early = ((col["Early<15-24>"]/col.sum())*100).round(2)
prime = ((col["Prime<25-54>"]/col.sum())*100).round(2)
mature = ((col["Mature<55-64>"]/col.sum())*100).round(2)
elderly = ((col["Elderly<65+>"]/col.sum())*100).round(2)
# A table to show the information
index = ['Children <0-14>','Early<15-24>','Prime<25-54>','Mature<55-64>','Elderly<65+>']
table = pd.DataFrame({"Number of patients":[col["Children <0-14>"], col["Early<15-24>"], col["Prime<25-54>"], col["Mature<55-64>"], col["Elderly<65+>"]], "Percentage":[children, early, prime, mature, elderly]}, index=index)
table
# Patients that showed up and received sms count
show_sms = t.query('sms_received == 1').sms_received.count()
show_sms
# Patients that showed up but did not receive sms
show_nosms = t.query('sms_received == 0').sms_received.count()
show_nosms
# Patients that did not show up but received sms
noshow_sms = nt.query('sms_received == 1').sms_received.count()
noshow_sms
# Patients that did not show up and did not receive sms
noshow_nosms = nt.query('sms_received == 0').sms_received.count()
noshow_nosms
# proportion of patients that received sms and showed up
p_show_sms = (show_sms/(show_sms+show_nosms)).round(2)
p_show_sms
## proportion of patients with  no_show = 1 and sms_received = 0
p_show_nosms = (show_nosms/(show_sms+show_nosms)).round(2)
p_show_nosms
# proportion of patients that no_show = 0 and sms_received = 1
p_noshow_sms = (noshow_sms/(noshow_sms+noshow_nosms)).round(2)
p_noshow_sms
## Proportion of no show = 0 and sms_received = 0
p_noshow_nosms = (noshow_nosms/(noshow_sms+noshow_nosms)).round(2)
p_noshow_nosms
# Table showing the percentage of sms received and not receive for patients that showed up and did not show up.
t_sms = pd.DataFrame({"sms":['Received','Not_receive'], "Showed_up":[p_show_sms, p_show_nosms], "No_show_up":[p_noshow_sms, p_noshow_nosms]})
t_sms
# Using Matplotlib
# .reset_index() helps in the use of index as x-axis


plt.figure(figsize=(8,7))
sns.barplot(data=tab.reset_index(), x=tab.index[:-1], y=tab['Total for each day'][:-1])
plt.xlabel('Days of the week')
plt.ylabel('Total number of patients')
plt.title('Chart showing the numbers of patients on each day of the week')
plt.show()
# The waiting time values for each category is placed in a list as waiting days.

waiting_days = [abs_wait_time[:-1].values, abs_nwait_time[:-1].values]

# Boxplot

labels=['Waiting time for show up','Waiting time for no show up']
fig1, ax1 = plt.subplots()
ax1.set_ylabel('Waiting days')
ax1.set_title('Waiting days for both that showed up and did not show up for the medical appointment')
color = ['blue','tan']
box = ax1.boxplot(waiting_days, widths=0.7, patch_artist=True, showmeans=True, showfliers=False, labels=labels);

for patch, color in zip(box['boxes'], color):
    patch.set_facecolor(color)
plt.show()
# The waiting day values for each  gender category is placed in a list as waiting days.

gender_waiting_days = [male_wait_time[:-1].values, female_wait_time[:-1].values]

# Boxplot
labels=['Male waiting time','female waiting time']
fig1, ax1 = plt.subplots()
ax1.set_ylabel(' Gender Waiting days')
ax1.set_title(' Boxplot showing the waiting days for each gender')
color = ['black','cyan']
box = ax1.boxplot(gender_waiting_days, widths=0.7, patch_artist=True, showmeans=True, showfliers=False, labels=labels);

for patch, color in zip(box['boxes'], color):
    patch.set_facecolor(color)
plt.show()
# Pie chart showing the age brackets

labels = ['Children', 'Early', 'Prime', 'Mature', 'Elderly']
sizes = [children, early, prime, mature, elderly]
explode = [0,0,0.1,0,0]
colors=['#E6A0B2','#7EBEBF','#82C292','#B3B968','#E6A667']# using HEX for the color
fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title('Percentages of age brackets that showed up most')

plt.show();
# Multiple bar plots using matplotlib.

plt.figure(figsize=(8,7))
x = np.arange(2)
width = 0.25
plt.bar(x+0, t_sms["Showed_up"], width=width, label="SMS" )
plt.bar(x+0.2, t_sms["No_show_up"], width=width, label="NO SMS")
plt.xlabel("SMS status for patients that showed up and patients that did not show up")
plt.ylabel("Proportion")
plt.title("Chart showing the proportion of sms")
plt.legend()
plt.show()
