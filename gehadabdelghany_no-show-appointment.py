import os

print(os.listdir("../input"))
# import packages

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

% matplotlib inline
# Load data and explore it.

df = pd.read_csv("../input/kagglev3/KaggleV2-May-2016.csv")

#print out a few lines

df.head()
df.columns
df.info()
df.shape
#look for missing vaules

df.isnull().sum()
#look for dublicated data

df.duplicated().sum()
df.describe()
df.columns = df.columns.str.lower()
df.columns
column_id = ['patient_id', 'appointment_id', 'gender', 'scheduled_day',

       'appointment_day', 'age', 'neighbourhood', 'scholarship', 'hipertension',

       'diabetes', 'alcoholism', 'handcap', 'sms_received', 'no_show']

df.columns = column_id

# for i in column_id:

#     for i in column_id:

#         column_id = "_".join(column_id)

# column_id

df.columns
# column_day = ['scheduledday','appointmentday']

# for i in column_day:

#     df[i] = df[i].apply(lambda x: x[:-3] +  x[-3:])

# df
column_day = ['scheduled_day','appointment_day']

for date in column_day:

    df[date] = pd.to_datetime(df[date])

df
df.info()
df["appointment_id"] = df["appointment_id"].astype(int)

df["appointment_id"]
df["patient_id"] = df["patient_id"].astype(int)
df
df['age'] = df['age'].astype(int)

df
df["no_show"].nunique()
df["no_show"] = np.where((df["no_show"]=="Yes"), 0, 1)

df["no_show"]
df["gender"] = np.where((df["gender"]=="F"), 1, 0)

df["gender"]
df
df.info()
no_shows = df.query('no_show == "0"')

no_shows
shows = df.query('no_show == "1"')

shows
#Number of appearing people  

Nshows= sum(df["no_show"] == 1)

Nshows
#Number of not attending people

Nno_shows= sum(df["no_show"] == 0)

Nno_shows
# Data to plot

labels = ['No Shows', 'Shows']

sizes = [Nno_shows, Nshows]

colors = [ 'tomato' ,'lightblue']

explode = (0, 0.1)



# The plot

plt.pie(sizes, explode=explode, labels=labels, colors=colors,

        autopct='%1.1f%%', shadow=True, startangle=40)

plt.title('Attendance percentage')

plt.axis('equal')

fig = plt.gcf()

fig.set_size_inches(5,5);
Nfemales = df[df["gender"] == 1]

Nfemales
# Number of all females

F = Nfemales['gender'].sum()

F
Nmales = df[df["gender"] == 0]

Nmales
# Number of all males

M = Nmales['gender'].count()

M
# Data to plot

labels = ['ALL Female', 'All Male']

sizes = [F, M]

colors = ['pink', 'royalblue']

explode = (0, 0.1)



# The plot

plt.pie(sizes, explode=explode, labels=labels, colors=colors,

        autopct='%1.1f%%', shadow=True, startangle=40)

plt.title('Gender Distribution by Shows')

plt.axis('equal')

fig = plt.gcf()

fig.set_size_inches(5,5);
female_shows = shows[shows['gender']==1]

female_shows
f = female_shows["gender"].sum()

f
males_shows = shows[shows['gender']==0]

males_shows
m = males_shows['gender'].count()

m
# Data to plot

labels = ['Female', 'male']

sizes = [f, m]

colors = [ 'tomato' ,'lightblue']

explode = (0, 0.1)



# The plot

plt.pie(sizes, explode=explode, labels=labels, colors=colors,

        autopct='%1.1f%%', shadow=True, startangle=40)

plt.title('Gender appointments')

plt.axis('equal')

fig = plt.gcf()

fig.set_size_inches(5,5);
female_no_shows = no_shows[no_shows['gender']==1]

female_no_shows
no_f = female_no_shows['gender'].sum()

no_f
males_no_shows = no_shows[no_shows['gender']==0]

males_no_shows
no_m = males_no_shows['gender'].count()

no_m
# Data to plot

labels = ['Female', 'male']

sizes = [no_f, no_m]

colors = ['lightblue', 'lightgreen']

explode = (0, 0.1)



# The plot

plt.pie(sizes, explode=explode, labels=labels, colors=colors,

        autopct='%1.1f%%', shadow=True, startangle=40)

plt.title('Gender Distribution by No Shows')

plt.axis('equal')

fig = plt.gcf()

fig.set_size_inches(5,5);
all_age_mean = df['age'].mean()

all_age_mean
age_show_mean = shows['age'].mean()

age_show_mean
age_no_show_mean = no_shows['age'].mean()

age_no_show_mean
# Data to plot 

showAge = shows['age']

noshowAge = no_shows['age']
## The plot of show data

plt.hist(showAge, bins=100)

plt.title('Age Distribution by Shows')

plt.xlabel('Age')

plt.ylabel('Number of Appointments')
## Plot no show data 

plt.hist(noshowAge, bins=100, color= "pink")

plt.title('Age Distribution by no Shows')

plt.xlabel('Age')

plt.ylabel('Number of Appointments')
df["neighbourhood"]
hood_show = shows.groupby('neighbourhood')['no_show'].count().reset_index(name= "Count").sort_values("Count")

hood_show
# The first five neighbourhoods that it is most likly to show in appointments

hood_show.tail()
# The neighbourhood that it is most likly to show in appointments

hood_show[hood_show["Count"] == hood_show["Count"].max()] 
hood = hood_show["neighbourhood"].head(10)

hood
cHood = hood_show["Count"].head(10)

cHood
## The plot of show data 

fig = plt.figure()

ax = fig.add_axes([0,0,4,3])

ax.bar(hood, cHood)
hood_no_show = no_shows.groupby('neighbourhood')['no_show'].count().reset_index(name= "Count").sort_values("Count")

hood_no_show
# The first five neighbourhoods that it is most likly to no show in appointments

hood_no_show.tail()
# The neighbourhood that it is most likly to no show in appointments

hood_no_show[hood_no_show["Count"] ==hood_no_show["Count"].max()] 
no_hood = hood_no_show["neighbourhood"].head(10)

no_cHood = hood_no_show["Count"].head(10)
## The plot of noshow data 

fig = plt.figure()

ax = fig.add_axes([0,0,4,3])

ax.bar(no_hood, no_cHood)
d = df.groupby("scholarship")["no_show"].count().reset_index(name= "Count")

d.head()
att = shows.groupby("scholarship")["no_show"].count().reset_index(name= "Count")

att
not_att = no_shows.groupby("scholarship")["no_show"].count().reset_index(name= "Count")

not_att
df.head()
#Total Hipertension

df_Hip1 = df[df["hipertension"]==1]

df_Hip1
df_Hip1["hipertension"].count()
#Total Diabetes

df_dia1 = df[df["diabetes"]==1]

df_dia1
df_dia1["diabetes"].count()
#Total Alcholism

df_alc1 = df[df["alcoholism"]==1]

df_alc1
df_alc1["alcoholism"].count()
#Total handicap 

df_hand = df[df["handcap"]==1] 

df_hand
df["handcap"].count()
# hipertension

df_hip2 = shows[shows["hipertension"]==1]

df_hip2
# diabetes

df_dia2 = shows[shows["diabetes"]==1]

df_dia2
#alcoholism

df_alch2 = shows[shows["alcoholism"]==1]

df_alch2
# Handicap

df_hand2 = shows[shows["handcap"]==1]

df_hand2
# The plot

labels = ['Hypertension','Diabetes','Alcoholism', 'Handicap']

sizes = [df_hip2.shape[0], df_dia2.shape[0], df_alch2.shape[0], df_hand2.shape[0]]

colors = ['palevioletred', 'lightpink', 'lavender', 'plum']

explode = (0, 0, 0.1, 0)



plt.pie(sizes, explode=explode, labels=labels, colors=colors,

        autopct='%1.1f%%', shadow=True, startangle=70)



plt.title('Health Designation by Shows')

plt.axis('equal')

fig = plt.gcf()

fig.set_size_inches(5,5)
# hipertension

df_hip3 = no_shows[no_shows["hipertension"]==1]

df_hip3
# diabetes

df_dia3 = no_shows[no_shows["diabetes"]==1]

df_dia3
#alcoholism

df_alch3 = no_shows[no_shows["alcoholism"]==1]

df_alch3
# Handicap

df_hand3 = no_shows[no_shows["handcap"]==1]

df_hand3
# The plot

labels = ['Hypertension','Diabetes','Alcoholism', 'Handicap']

sizes = [df_hip3.shape[0], df_dia3.shape[0], df_alch3.shape[0], df_hand3.shape[0]]

colors = ['steelblue', 'lightblue', 'lavender', 'turquoise']

explode = (0, 0, 0.1, 0)



plt.pie(sizes, explode=explode, labels=labels, colors=colors,

        autopct='%1.1f%%', shadow=True, startangle=70)



plt.title('Health Designation by no Show')

plt.axis('equal')

fig = plt.gcf()

fig.set_size_inches(5,5)
# Total number of patients that recived SMS 

smsNo = df[df["sms_received"] == 1] 

smsNo
smsNo.shape[0]
sms_show = shows[shows["sms_received"] == 1] 

sms_show
a = sms_show.shape[0]

a
sms_no_show = no_shows[no_shows["sms_received"] == 1] 

sms_no_show
b = sms_no_show.shape[0]

b
## The plot of SMS data 

locations = [1, 2]

heights = [a, b]

labels = ['Shows', 'No-Shows']



bar1 = plt.bar(locations, heights, tick_label=labels, color=['slateblue','darkslateblue'])

plt.title('SMS Messages Received')

plt.xlabel('Appointments')

plt.ylabel('SMS Receipt Rate');

no_sms_show = shows[shows["sms_received"] == 0] 

no_sms_show
x = no_sms_show.shape[0]

x
no_sms_no_show = no_shows[no_shows["sms_received"] == 0] 

no_sms_no_show
y = no_sms_no_show.shape[0]

y
## The plot of SMS data 

locations = [1, 2]

heights = [a, b]

labels = ['Shows', 'No-Shows']



bar1 = plt.bar(locations, heights, tick_label=labels, color=['lightblue','pink'])

plt.title('SMS Messages not Received')

plt.xlabel('Appointments')

plt.ylabel('SMS Receipt Rate');

from subprocess import call

call(['python', '-m', 'nbconvert', 'Investigate_a_Dataset.ipynb'])