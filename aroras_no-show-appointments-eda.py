import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
%matplotlib inline
# Load CSV and create a dataframe df
df = pd.read_csv('../input/noshowappointments/KaggleV2-May-2016.csv')
# no of samples and columns in the dataset
df.shape
# Datatypes of columns
df.info()
# initial look at the data
df.head(5)
# gather descriptive statistics about data
df.describe()
# finding duplicate rows in the dataset
df.duplicated().sum()
# looking at unique count in columns
df.nunique()
# Checking for any outliers in age
df['Age'].unique()
# checking Neighbourhood
df['Neighbourhood'].unique()
# unique values in Gender and Scholarship
print("Gender: {} Scholarship: {}".format(df['Gender'].unique(), df['Scholarship'].unique()))
# unique values in Hipertension and Diabetes
print("Hipertension: {} Diabetes: {}".format(df['Hipertension'].unique(), df['Diabetes'].unique()))
# unique values in Alcoholism and Handcap
print("Alcoholism: {} Handcap: {}".format(df['Alcoholism'].unique(), df['Handcap'].unique()))
# unique values SMS_received and No-show
print("SMS_received: {} No-show: {}".format(df['SMS_received'].unique(), df['No-show'].unique()))
# rename Handcap to Handicap, Hipertension to Hypertension and No-show to No_show
df = df.rename(columns = {'Handcap': 'Handicap', 'Hipertension': 'Hypertension','No-show': 'No_show'})
# check the columns are renamed
df.head(1)
# Patientid and AppointmentID columns are not used for analysis
df.drop(['PatientId', 'AppointmentID'], axis=1, inplace=True)
# confirm the columns are dropped
df.head(1)
# find number of rows where Age is between 0 and 100 (both inclusive) 
len(df.query("Age >=0 and Age <=100")), len(df)
# drop ouliers in data
df = df.query("Age >=0 and Age <=100")
# check the rows are removed
len(df.query("Age < 0 and Age > 100"))
# define a function to extract date
# function returns date object from datetime data passed as argument
def extract_date(d):
    return pd.to_datetime(d).date()
# extract date from ScheduledDay keeping same yyyy-MM-dd date format
df['ScheduledDay'] = df['ScheduledDay'].apply(extract_date)
# extract date from AppointmentDay
df['AppointmentDay'] = df['AppointmentDay'].apply(extract_date)
df.head(1)
df.dtypes
# convert object to datetime for ScheduleDay
df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'])
# convert object to datetime for AppointmentDay
df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'])
# verify both columns ScheduledDay and AppointmentDay are of type datetime
df.info()
# check if scheduled date is after appointment date
outliers_date = (df['AppointmentDay'] - df['ScheduledDay']).dt.days
# extract only outliers rows where date is < 0
outliers_date = outliers_date[outliers_date < 0]
# find size and row indexes
outliers_date.index, outliers_date.index.size
# removing 5 rows as they're treated as outliers or human error
df.drop(outliers_date.index, inplace=True)
df.shape
# count of each value in Handicap Data
df.Handicap.value_counts()
# converting [1,2,3,4] to 1 
df['Handicap'] = (df.Handicap > 0).astype(np.int)
# count values to confirm numbers match
df.Handicap.value_counts()
# calculating diffrence and converting to number of days
df['Waiting_Time'] = (df['AppointmentDay'] - df['ScheduledDay']).dt.days
# verify Waiting_Time doesn't contains any negative values.
df['Waiting_Time'].min(), df['Waiting_Time'].max()
# check new column Waiting_Time is created 
df.head()
# create weekday dictionary for mapping numeric to weekday
mapDayOfWeek={0:'Monday', 1:'Tuesday', 2:'Wednesday', 3:'Thursday', 4:'Friday', 5:'Saturday', 6:'Sunday'}
# create new column Weekday to store weekday
df['Weekday'] = df['AppointmentDay'].dt.dayofweek.map(mapDayOfWeek)
# check the new column Weekday is created with correct values
df.head(1)
df.shape
# Bin edges that will be used to "cut" the data into groups
# we're dividing the waiting time into 5 groups based on number of days (waiting time)
bin_edges = [-1, 0, 7, 30, 90, 180 ]
# create bin names
bin_names = ['same day', '1 week', '1 month', '3 months', '6 months+']
# create column Waiting_Period
df['Waiting_Period'] = pd.cut(df['Waiting_Time'], bin_edges, labels=bin_names)
# verify Waiting_Period created
df.head(1)
# Data distribution of numeric data
df.hist(figsize=(10,10));
# Patient appointments by Gender
ax = df['Gender'].value_counts().plot(kind='bar');
ax.set_title('Appointments for Female/Male' )
ax.set_xticklabels(['Female', 'Male'], rotation=0);
ax.set_xlabel('Gender')
ax.set_ylabel('Count')
plt.show;
# lets see the distribution of Waiting_Period
ax = df.Waiting_Period.hist();
ax.set_title('Waiting Periods')
ax.set_xlabel('Waiting Period')
ax.set_ylabel('Count')
plt.show;
# declare the sort order for weekdays
week_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
# Weekday plot against show/No Show
plt.figure(figsize=(12,5))
ax = sns.countplot(x=df.Weekday, hue=df.No_show, order = week_order)
ax.set_title("Show/NoShow on Weekdays")
plt.show()
# count plot for patients having Hypertension
ax = df['Hypertension'].value_counts().plot(kind='bar', title='Patients with Hypertenstion');
ax.set_xticklabels(['No', 'Yes'], rotation=0);
ax.set_xlabel('Hypertension')
ax.set_ylabel('Count')
plt.show;
# count plot for SMS received
ax = df['SMS_received'].value_counts().plot(kind='bar', title='SMS Received By Patients');
ax.set_xticklabels(['No', 'Yes'], rotation=0);
ax.set_xlabel('SMS received')
ax.set_ylabel('Count')
plt.show;
# get counts by Gender type
df['Gender'].value_counts()
# check the count for each Gender for show/no_show
pd.crosstab(df.Gender, df.No_show, margins=True, margins_name="Total")
# Let's plot Show/No_Show appointments for Males and Females
ax = sns.countplot(x=df.Gender, hue=df.No_show, data=df)
ax.set_xticklabels(['Female', 'Male'])
ax.set_title('Show/No Show for Males and Females')
plt.show()
# let's check the proportion of Females for No show
df.query("Gender == 'F' and No_show == 'Yes'").size / df.query("Gender == 'F'").size
# let's check the proportion of Males for No show
df.query("Gender == 'M' and No_show == 'Yes'").size / df.query("Gender == 'M'").size
# let's look at the count of patients making appointments w.r.t Age
ax = df.Age.value_counts().plot(kind='bar', figsize=(18,5));
ax.set_title('Appointments By Age')
ax.set_xlabel('Age')
ax.set_ylabel('Count')
plt.show();
# check the count for top 5 Age visitors
df.Age.value_counts().nlargest(5)
# Let's plot Age with Show/NoShow
plt.figure(figsize=(18,5))
ax = sns.countplot(x=df.Age, hue=df.No_show)
ax.set_title("Show/NoShow Appointments by Age")
plt.show()
# let's check the proportion of Age with 'no show'(Do not show for appointments)
no_show_prop = df.query("No_show == 'Yes'").groupby(['Age']).size() / df.groupby(['Age']).size()
# let's check the proportion of Age who showed up for appointments
show_prop = df.query("No_show == 'No'").groupby(['Age']).size()/df.groupby(['Age']).size()
df_age_prop = pd.DataFrame({'Show': show_prop,'No Show': no_show_prop})
ax = df_age_prop.plot.line()
ax.set_title('Show/No Show Comparison By Age')
ax.set_xlabel('Age')
ax.set_ylabel('Count')
plt.show();
# check the counts for Waiting_Period for show/no_show
pd.crosstab(df.Waiting_Period, df.No_show, margins=True, margins_name="Total")
# plot waiting period against Show/No Show
plt.figure(figsize=(12,5))
ax = sns.countplot(x=df.Waiting_Period, hue=df.No_show)
ax.set_title("Show/NoShow Waiting Period")
ax.set_xlabel('Waiting Period')
plt.show();
# calculate percentage Waiting Period for Patients who showed up
show_wtg_prop = df.query("No_show == 'No'").groupby(['Waiting_Period']).size()/df.groupby(['Waiting_Period']).size()
# calculate percentage Waiting Period for Patients who DO NOT show up
no_show_wtg_prop = df.query("No_show == 'Yes'").groupby(['Waiting_Period']).size()/df.groupby(['Waiting_Period']).size()
# Let's plot both to get a pattern 
df_wtg_prop = pd.DataFrame({'Show': show_wtg_prop,'No Show': no_show_wtg_prop})
ax = df_wtg_prop.plot.line()
ax.set_title('Show/NoShow Waiting Period')
ax.set_xlabel('Waiting Period')
ax.set_ylabel('Count')
plt.show();
no_show_wk_prop = df.query("No_show == 'Yes'").groupby(['Weekday']).size()/df.groupby(['Weekday']).size()
no_show_wk_prop.sort_values(ascending=False)
# Let's check the percentage of Hypertension Patients with show/No_show
pd.crosstab(df.Hypertension, df.No_show, normalize = "index")
ax = pd.crosstab(df.Hypertension, df.No_show, normalize = "index").plot(kind='bar', title='Show/No Show with Hypertension');
ax.set_xticklabels(['No', 'Yes'], rotation=0);
ax.set_xlabel('Hypertension')
ax.set_ylabel('Ratio')
plt.show();
# lets check the proportion of SMS received with Show/No Show
pd.crosstab(df.SMS_received, df.No_show, normalize = "index")
ax = pd.crosstab(df.SMS_received, df.No_show, normalize = "index").plot(kind='bar');
ax.set_xticklabels(['No', 'Yes'], rotation=0);
ax.set_title('SMS Received By Patients')
ax.set_xlabel('SMS Received')
ax.set_ylabel('Ratio')
plt.show();
