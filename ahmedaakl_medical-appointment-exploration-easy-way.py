
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skew,norm
import numpy as np

%matplotlib inline
def draw_countfig(data,x,hue=None, figs_num=1, figsize=(20,5), title1="", title2="",rotation = 90,order=None, legend_loc='upper right'):
    if figs_num == 2:
        f, ax = plt.subplots(1,2, figsize=(20,5))
        ax1 = sns.countplot(x=data[x], order=order, ax=ax[0]);
        ax2 = sns.countplot(x=x, hue=hue, data=data, order=order);
        plt.legend(loc='upper right')
        plt.xticks(rotation=rotation)
        ax1.title.set_text(title1)
        ax2.title.set_text(title2)
        plt.show()
    elif (figs_num==1) & (hue != None):
        plt.figure(figsize=figsize)
        ax = sns.countplot(x=data[x], data=data, hue=hue);        
        ax.set_title(title1)
        plt.legend(loc='upper right')
        plt.xticks(rotation=rotation)
        plt.show()
    elif figs_num==1:
        plt.figure(figsize=figsize)
        ax = sns.countplot(x=data[x], data=data);        
        ax.set_title(title1)
        plt.xticks(rotation=rotation)
        plt.show()
    

# Load your data and print out a few lines. Perform operations to inspect data
#   types and look for instances of missing or possibly errant data.

data = pd.read_csv("../input/noshowappointments/KaggleV2-May-2016.csv")

data.head()
data.tail()
# get general information about the dataset
data.info()

# get some basic descriptive statistics about the dataset
data.describe()

# check for duplicates
data.duplicated().sum()

data.columns
#Renaming columns
data.rename(columns = lambda x: x.strip().lower().replace("-", "_"), inplace=True)
# check columns name change
data.head()
# convert both scheduleday and appointmentday to date.
data['scheduledday'] = pd.to_datetime(data['scheduledday']).astype('datetime64[ns]')
data['appointmentday'] = pd.to_datetime(data['appointmentday'].astype('datetime64[ns]'))
#check if date changed from string into datetime type
data.info()
data['appointmentid'].nunique()
data['patientid'].nunique()
data['age'].unique()
# patient with age less than 0
print("Number of patients with age less than zero: ", data[data['age'] <0].shape[0])
# patient with age equal to 0
print("Number of patients with age less than zero: ", data[data['age'] == 0].shape[0])
# patient with age larger than 100
print("Number of patients with age larger than 100: ",data[data['age'] > 100].shape[0])
# drop patient with -1 year old.
data.drop(data[data['age'] == -1].index, axis=0, inplace=True)
# check for negative values
data['age'].unique()
# exploring the distribution of the target variable (no-show)
draw_countfig(data,'no_show',figs_num=1, title1='Patients show and no_show count',figsize=(6,4))
# number of patients that missed the appointment
print("Number of patients that missed the appointments is: ", data[data['no_show'] == 'Yes'].shape[0],
      "Number of patients that have the appointments is: ", data[data['no_show'] == 'No'].shape[0])
f,ax = plt.subplots(1,2,figsize=(16,5))
sns.distplot(data['age'],fit=norm,ax=ax[0], bins=20)
sns.boxplot(data['age'])
plt.show()
draw_countfig(data,'age','no_show',rotation=90, title1= "Age Show / No show patients")
# bin the age with base 5 
bin_ranges = np.arange(0,data['age'].max(),5)

data['age_bins'] = pd.cut(data['age'],bin_ranges)
draw_countfig(data,x='age_bins',hue='no_show')
fig_title = "Number of Patients based on Gender"
draw_countfig(data,x='gender', figsize=(6,4), rotation=0,title1=fig_title)
print("Number of Female patients: ", data[data['gender'] == 'F'].shape[0],
      "Number of Male patients: ", data[data['gender'] == 'M'].shape[0])
fig_title = "Show / No show patients based on Gender"
draw_countfig(data,x='gender', hue='no_show', title1=fig_title, figsize=(8,5))
data.groupby('no_show')['gender'].value_counts()
draw_countfig(data,x='age_bins',hue='gender')
data.columns
# get the name of the day from both schedule and appointment days
data['scheduledday_name'] = data['scheduledday'].dt.day_name()
# data['appointmentday_name'] = data['appointmentday'].dt.day_name()

week_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
# plot a count plot for schedule day and its relation with no_show
title1 = 'Number of schedules in per week'
title2 = 'Relationship between schedule day and show up'
draw_countfig(data,x='scheduledday_name', hue='no_show',title1= title1, title2=title2,figs_num=2)

title1 = "Number of scholarships for patients"
title2 = "Show / No show based on Scholarship"
draw_countfig(data, x='scholarship',hue='no_show',figs_num=2, title1=title1, title2=title2)

data.groupby('no_show')['scholarship'].value_counts()
draw_countfig(data,x='age_bins',hue='scholarship')

data.tail(10)
data.appointmentday.dt.month_name().unique()
