import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder
plt.style.use('fivethirtyeight')
%matplotlib inline
interview_df = pd.read_csv('../input/Interview.csv', delimiter=',')
interview_df
interview_df.drop(['Unnamed: 23', 'Unnamed: 24', 'Unnamed: 25', 'Unnamed: 26', 'Unnamed: 27', 'Name(Cand ID)'], 
                  axis=1, inplace=True)
interview_df.info()
interview_df.columns = ['Date', 'Client', 'Industry', 'Location', 'Position', 'Skillset', 'Interview_Type', 
                     'Gender', 'Curr_Location', 'Job_Location', 'Venue', 'Native_Loc', 'Necc_Perm', 'No_random_meet', 
                      'Call_3_hours', 'Alternative_Number', 'Printout_resume', 'Details_Clear_Landmark', 
                      'Shared_Letter', 'Exp_Attendance', 'Ob_Attendance', 'Martial_Status']
interview_df.head()
interview_df.tail()
interview_df.drop(1233, inplace=True)
interview_df.info()
for column in interview_df.columns:
    print(column, interview_df[column].unique())
    print('-'*40)
interview_df.drop(['Date', 'Skillset', 'Native_Loc', 'Alternative_Number'], axis=1, inplace=True)
interview_df.Client.replace(['Aon Hewitt', 'Hewitt', 'Aon hewitt Gurgaon'], 'Hewitt', inplace=True)
interview_df.Client.replace(['Standard Chartered Bank', 'Standard Chartered Bank Chennai'], 
                            'Standard Chartered', inplace=True)

interview_df.Industry.replace(['IT Services', 'IT Products and Services', 'IT'], 
                              'IT', inplace=True)

interview_df.Location.replace(['chennai', 'Chennai', 'chennai ', 'CHENNAI'], 'Chennai', inplace=True)
interview_df.Location.replace('- Cochin- ', 'Cochin', inplace=True)
interview_df.Location.replace('Gurgaonr', 'Gurgaon', inplace=True)

interview_df.Curr_Location.replace(['chennai', 'Chennai', 'chennai ', 'CHENNAI'], 'Chennai', inplace=True)
interview_df.Curr_Location.replace('- Cochin- ', 'Cochin', inplace=True)

interview_df.Job_Location.replace('- Cochin- ', 'Cochin', inplace=True)

interview_df.Venue.replace('- Cochin- ', 'Cochin', inplace=True)

interview_df.Exp_Attendance.fillna('yes', inplace=True)
interview_df.Exp_Attendance.replace(['Yes', '10.30 Am', '11:00 AM'], 'yes', inplace=True)
interview_df.Exp_Attendance.replace(['No', 'NO'], 'no', inplace=True)

interview_df.Ob_Attendance.replace(['Yes', 'yes '], 'yes', inplace=True)
interview_df.Ob_Attendance.replace(['No', 'NO', 'No ', 'no '], 'no', inplace=True)

interview_df.Printout_resume.replace('Yes', 'yes', inplace=True)
interview_df.Printout_resume.replace(['No', 'Not yet', 'na', 'Na', 'Not Yet', 
                                      'No- will take it soon'], 'no',inplace=True)

interview_df.Necc_Perm.replace(['Yes', 'yes'], 'yes', inplace=True)
interview_df.Necc_Perm.replace(['No', 'Not yet', 'NO', 'Na', 'Yet to confirm'] ,'no', inplace=True)

interview_df.Details_Clear_Landmark.replace(['Yes', 'yes'], 'yes', inplace=True)
interview_df.Details_Clear_Landmark.replace(['No', 'na', 'no', 'Na', 'No- I need to check'] ,'no', inplace=True)

interview_df.Interview_Type.replace(['Scheduled Walk In', 'Scheduled ', 'Scheduled Walk In', 'Sceduled walkin'],
                                    'Scheduled', inplace=True)
interview_df.Interview_Type.replace(['Walkin '], 'Walkin', inplace=True)

interview_df.No_random_meet.replace(['Yes', 'yes'], 'yes', inplace=True)
interview_df.No_random_meet.replace(['Na', 'No', 'Not Sure', 'cant Say', 'Not sure'] ,'no', inplace=True)

interview_df.Call_3_hours.replace(['No', 'No Dont', 'Na'], 'no', inplace=True)
interview_df.Call_3_hours.replace(['Yes', 'yes'], 'yes', inplace=True)

interview_df.Shared_Letter.replace(['Yes', 'yes'], 'yes', inplace=True)
interview_df.Shared_Letter.replace(['Havent Checked', 'No', 'Need To Check', 'Not sure', 
                                   'Yet to Check','Not Sure', 'Not yet', 'no', 'na', 'Na'], 'no', inplace=True)
interview_df.info()
sns.countplot(x=interview_df.Printout_resume, hue=interview_df.Exp_Attendance)
index_nan_print = interview_df['Printout_resume'][interview_df['Printout_resume'].isnull()].index
for i in index_nan_print:
    
    if interview_df.iloc[i]['Exp_Attendance'] == 'no':
        interview_df.iloc[i]['Printout_resume'] = 'no'
    else:
        interview_df.iloc[i]['Printout_resume'] = 'yes'
sns.countplot(x=interview_df.Necc_Perm, hue=interview_df.Exp_Attendance)
plt.show()
sns.countplot(x=interview_df.Necc_Perm, hue=interview_df.Ob_Attendance)
index_nan_perm = interview_df['Necc_Perm'][interview_df['Necc_Perm'].isnull()].index
for i in index_nan_perm:
    
    if interview_df.iloc[i]['Exp_Attendance'] == 'no':
        interview_df.iloc[i]['Necc_Perm'] = 'no'
    else:
        interview_df.iloc[i]['Necc_Perm'] = 'yes'
sns.countplot(x=interview_df.Details_Clear_Landmark, hue=interview_df.Exp_Attendance)
plt.show()
sns.countplot(x=interview_df.Details_Clear_Landmark, hue=interview_df.Ob_Attendance)
index_nan_details = interview_df['Details_Clear_Landmark'][interview_df['Details_Clear_Landmark'].isnull()].index
for i in index_nan_details:
    
    if (interview_df.iloc[i]['Exp_Attendance'] == 'no'):
        interview_df.iloc[i]['Details_Clear_Landmark'] = 'no'
    else:
        interview_df.iloc[i]['Details_Clear_Landmark'] = 'yes'
sns.countplot(x=interview_df.No_random_meet, hue=interview_df.Exp_Attendance)
plt.show()
sns.countplot(x=interview_df.No_random_meet, hue=interview_df.Ob_Attendance)
index_nan_details = interview_df['No_random_meet'][interview_df['No_random_meet'].isnull()].index
for i in index_nan_details:
    
    if (interview_df.iloc[i]['Ob_Attendance'] == 'no' and interview_df.iloc[i]['Exp_Attendance'] == 'no'):
        interview_df.iloc[i]['No_random_meet'] = 'no'
    else:
        interview_df.iloc[i]['No_random_meet'] = 'yes'
sns.countplot(x=interview_df.Shared_Letter, hue=interview_df.Exp_Attendance)
plt.show()
sns.countplot(x=interview_df.Shared_Letter, hue=interview_df.Ob_Attendance)
index_nan_details = interview_df['Shared_Letter'][interview_df['Shared_Letter'].isnull()].index
for i in index_nan_details:
    
    if (interview_df.iloc[i]['Ob_Attendance'] == 'no' or interview_df.iloc[i]['Exp_Attendance'] == 'no'):
        interview_df.iloc[i]['Shared_Letter'] = 'no'
    else:
        interview_df.iloc[i]['Shared_Letter'] = 'yes'
sns.countplot(x=interview_df.Call_3_hours, hue=interview_df.Exp_Attendance)
plt.show()
sns.countplot(x=interview_df.Call_3_hours, hue=interview_df.Ob_Attendance)
index_nan_details = interview_df['Call_3_hours'][interview_df['Call_3_hours'].isnull()].index
for i in index_nan_details:
    
    if (interview_df.iloc[i]['Exp_Attendance'] == 'no'):
        interview_df.iloc[i]['Call_3_hours'] = 'no'
    else:
        interview_df.iloc[i]['Call_3_hours'] = 'yes'
interview_df.info()
f, (ax1, ax2) = plt.subplots(1,2, figsize=(15,8))
sns.countplot(x=interview_df.Exp_Attendance, ax=ax1)
ax1.set_title('Expected attendance')
sns.countplot(x=interview_df.Ob_Attendance, ax=ax2)
ax2.set_title('Observed attendance')
f, (ax1, ax2) = plt.subplots(2, figsize=(15,16))
sns.countplot(x=interview_df.Position, hue=interview_df.Exp_Attendance, ax=ax1)
sns.countplot(x=interview_df.Position, hue=interview_df.Ob_Attendance, ax=ax2)
f, (ax1, ax2) = plt.subplots(1,2, figsize=(15,8))
sns.countplot(x=interview_df.Job_Location, hue=interview_df.Exp_Attendance, ax=ax1)
sns.countplot(x=interview_df.Job_Location, hue=interview_df.Ob_Attendance, ax=ax2)
f, (ax1, ax2) = plt.subplots(2, figsize=(15,16))
sns.countplot(x=interview_df.Location, hue=interview_df.Client, ax=ax1)
ax1.legend(loc='right')
sns.countplot(x=interview_df.Job_Location, hue=interview_df.Client, ax=ax2)
f, (ax1, ax2) = plt.subplots(1,2, figsize=(15,8))
sns.countplot(x=interview_df.Curr_Location, hue=interview_df.Exp_Attendance, ax=ax1)
sns.countplot(x=interview_df.Job_Location, hue=interview_df.Exp_Attendance, ax=ax2)
f, (ax1, ax2) = plt.subplots(1,2, figsize=(15,8))
sns.countplot(x=interview_df.Curr_Location, hue=interview_df.Ob_Attendance, ax=ax1)
sns.countplot(x=interview_df.Job_Location, hue=interview_df.Ob_Attendance, ax=ax2)
interview_df.groupby(['Curr_Location', 'Job_Location', 'Exp_Attendance', 'Ob_Attendance']).size()
f, ax1 = plt.subplots(1, figsize=(10,6))
sns.countplot(x=interview_df.Industry, ax=ax1)
interview_df.Industry.value_counts()
f, (ax1, ax2) = plt.subplots(2, figsize=(15,16))
sns.countplot(x=interview_df.Industry, hue=interview_df.Position, ax=ax1)
sns.countplot(x=interview_df.Job_Location, hue=interview_df.Industry, ax=ax2)
interview_df.groupby(['Industry', 'Position']).size()
f, (ax1, ax2) = plt.subplots(2, figsize=(15,16))
sns.countplot(x=interview_df.Interview_Type, hue=interview_df.Client, ax=ax1)
sns.countplot(x=interview_df.Position, hue=interview_df.Client, ax=ax2)
interview_df.groupby(['Industry', 'Client', 'Position']).size()
f, (ax1, ax2) = plt.subplots(2, figsize=(15,16))
sns.countplot(x=interview_df.Gender, hue=interview_df.Industry, ax=ax1)
sns.countplot(x=interview_df.Gender, hue=interview_df.Position, ax=ax2)
interview_df.groupby(['Gender', 'Industry', 'Position']).size()
f , ax = plt.subplots(1, figsize=(10,8)) 
sns.countplot(x=interview_df.Gender, hue=interview_df.Martial_Status, ax=ax)
interview_df.groupby(['Gender', 'Martial_Status', 'Exp_Attendance', 'Ob_Attendance']).size()
le = LabelEncoder()
interview_df = interview_df.apply(le.fit_transform)
interview_df.corr()
f , ax = plt.subplots(1, figsize=(15,10)) 
sns.heatmap(interview_df.corr(), ax=ax, annot=True)