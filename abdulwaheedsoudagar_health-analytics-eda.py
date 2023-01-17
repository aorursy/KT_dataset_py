# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import seaborn as sns
import matplotlib.pyplot as plt
Patient_Profile_df=pd.read_csv('/kaggle/input/healthcare-analytics/Train/Patient_Profile.csv')
First_Health_Camp_Attended_df=pd.read_csv('/kaggle/input/healthcare-analytics/Train/First_Health_Camp_Attended.csv')
Third_Health_Camp_Attended_df=pd.read_csv('/kaggle/input/healthcare-analytics/Train/Third_Health_Camp_Attended.csv')
Health_Camp_Detail_df=pd.read_csv('/kaggle/input/healthcare-analytics/Train/Health_Camp_Detail.csv')
test_df=pd.read_csv('/kaggle/input/healthcare-analytics/Train/test.csv')
Train_df=pd.read_csv('/kaggle/input/healthcare-analytics/Train/Train.csv')
Second_Health_Camp_Attended_df=pd.read_csv('/kaggle/input/healthcare-analytics/Train/Second_Health_Camp_Attended.csv')
def finding_outlier(dataframe,cols):
    plt.figure(figsize=(16,16))
    for i, col in enumerate(cols):
        plt.subplot(4,4,i+1)
        sns.boxplot(dataframe[col])
        plt.tight_layout()
def converting_datatype(series,type):
    return series.astype(type)
def replacing_none_values(series):
    return series.replace({None:0})
Patient_Profile_df.columns
Patient_Profile_df.shape
Patient_Profile_df.isnull().sum()/len(Patient_Profile_df) * 100
Patient_Profile_df['First_Interaction_month_year']=Patient_Profile_df['First_Interaction'].str[3:]
Patient_Profile_df['First_Interaction_year']=Patient_Profile_df['First_Interaction'].str[7:]
Patient_Profile_df['First_Interaction_year']=converting_datatype(Patient_Profile_df['First_Interaction_year'],str)
print("Number of patient ",Patient_Profile_df['Patient_ID'].nunique())
print("Percentage of patient who follows medcamp online -",Patient_Profile_df.Online_Follower.value_counts(normalize=True).values[1]*100)
print("Percentage of patient has shared details of a camp on his LinkedIn id -",Patient_Profile_df.LinkedIn_Shared.value_counts(normalize=True).values[1]*100)
print("Percentage of patient has tweeted about the health camp -",Patient_Profile_df.Twitter_Shared.value_counts(normalize=True).values[1]*100)
print("Percentage of patient has shared an update about the health camp Facebook -",Patient_Profile_df.Facebook_Shared.value_counts(normalize=True).values[1]*100)
fig, ax =plt.subplots(1,2,figsize=(19,5))
sns.countplot(data=Patient_Profile_df,x='Income',ax=ax[0])
sns.countplot(data=Patient_Profile_df,x='City_Type',ax=ax[1])
fig.show()
fig, ax =plt.subplots(1,2,figsize=(19,5))
sns.distplot(Patient_Profile_df[Patient_Profile_df['Education_Score']!='None']['Education_Score'],ax=ax[0],kde=False)
sns.distplot(Patient_Profile_df[Patient_Profile_df['Age']!='None']['Age'],ax=ax[1],kde=False)
fig.show()
fig, ax =plt.subplots(1,2,figsize=(19,5))
sns.boxplot(x=Patient_Profile_df[Patient_Profile_df['Education_Score']!='None']['First_Interaction_year'],\
            y=Patient_Profile_df[Patient_Profile_df['Education_Score']!='None']['Education_Score'].astype(float),ax=ax[0])
sns.boxplot(x=Patient_Profile_df[Patient_Profile_df['Age']!='None']['First_Interaction_year'],\
            y=Patient_Profile_df[Patient_Profile_df['Age']!='None']['Age'].astype(float),ax=ax[1])
fig.show()
Health_Camp_Detail_df.columns
Health_Camp_Detail_df.isnull().sum()
Health_Camp_Detail_df['Camp_Duration'] = pd.to_datetime(Health_Camp_Detail_df['Camp_End_Date'])-pd.to_datetime(Health_Camp_Detail_df['Camp_Start_Date'])
print("Number of health camps ",Health_Camp_Detail_df['Health_Camp_ID'].nunique())
print('Avg duration of a camp is ',Health_Camp_Detail_df['Camp_Duration'].mean())


First_Health_Camp_Attended_df.columns
First_Health_Camp_Attended_df.shape
First_Health_Camp_Attended_df.isnull().sum()/len(First_Health_Camp_Attended_df)*100
First_Health_Camp_Attended_df.drop(['Unnamed: 4'],axis=1,inplace=True)
n_patients_visted_firstcamp=First_Health_Camp_Attended_df[First_Health_Camp_Attended_df['Patient_ID'].isin(Patient_Profile_df['Patient_ID'].unique())]['Patient_ID'].nunique()
patients_visted_firstcamp=First_Health_Camp_Attended_df[First_Health_Camp_Attended_df['Patient_ID'].isin(Patient_Profile_df['Patient_ID'].unique())]['Patient_ID'].unique()
print('Number of Patients visited first camp ',n_patients_visted_firstcamp)
n_multiple_time_visit = First_Health_Camp_Attended_df['Patient_ID'].value_counts().reset_index(name="count").query("count > 1")['index'].nunique()
multiple_time_visit = First_Health_Camp_Attended_df['Patient_ID'].value_counts().reset_index(name="count").query("count > 1")['index'].unique()
print('Number of patients visited Multiple time ',n_multiple_time_visit)
fig, ax =plt.subplots(1,2,figsize=(19,5))
sns.distplot(First_Health_Camp_Attended_df['Donation'],ax=ax[0],kde=False)
sns.distplot(First_Health_Camp_Attended_df['Health_Score'],ax=ax[1],kde=False)
fig.show()
fig, ax =plt.subplots(1,2,figsize=(19,5))
sns.distplot(First_Health_Camp_Attended_df[First_Health_Camp_Attended_df['Patient_ID'].isin(multiple_time_visit)]['Health_Score'],ax=ax[0],kde=False).set_title('Multiple Visit')
sns.distplot(First_Health_Camp_Attended_df[~First_Health_Camp_Attended_df['Patient_ID'].isin(multiple_time_visit)]['Health_Score'],ax=ax[1],kde=False).set_title('Single Visit')
fig.show()
print("Avg health score of patients of Mutiple visit",First_Health_Camp_Attended_df[First_Health_Camp_Attended_df['Patient_ID'].isin(multiple_time_visit)]['Health_Score'].mean())
print("Avg health score of patients of single visit",First_Health_Camp_Attended_df[~First_Health_Camp_Attended_df['Patient_ID'].isin(multiple_time_visit)]['Health_Score'].mean())

Second_Health_Camp_Attended_df.columns
Second_Health_Camp_Attended_df.isnull().sum()/len(Second_Health_Camp_Attended_df)*100
n_patients_visted_firstcamp=Second_Health_Camp_Attended_df[Second_Health_Camp_Attended_df['Patient_ID'].isin(Patient_Profile_df['Patient_ID'].unique())]['Patient_ID'].nunique()
print('Number of Patients visited Second camp ',n_patients_visted_firstcamp)

n_multiple_time_visit = Second_Health_Camp_Attended_df['Patient_ID'].value_counts().reset_index(name="count").query("count > 1")['index'].nunique()
multiple_time_visit = Second_Health_Camp_Attended_df['Patient_ID'].value_counts().reset_index(name="count").query("count > 1")['index'].unique()
print('Number of patients visited Multiple time ',n_multiple_time_visit)
fig, ax =plt.subplots(1,2,figsize=(19,5))
sns.distplot(Second_Health_Camp_Attended_df[Second_Health_Camp_Attended_df['Patient_ID'].isin(multiple_time_visit)]['Health Score'],ax=ax[0],kde=False).set_title('Multiple Visit')
plt.title('add ')
sns.distplot(Second_Health_Camp_Attended_df[~Second_Health_Camp_Attended_df['Patient_ID'].isin(multiple_time_visit)]['Health Score'],ax=ax[1],kde=False).set_title('Single Visit')
fig.show()
print("Avg health score of patients of Mutiple visit",Second_Health_Camp_Attended_df[Second_Health_Camp_Attended_df['Patient_ID'].isin(multiple_time_visit)]['Health Score'].mean())
print("Avg health score of patients of single visit",Second_Health_Camp_Attended_df[~Second_Health_Camp_Attended_df['Patient_ID'].isin(multiple_time_visit)]['Health Score'].mean())
n_first_second_visit = Second_Health_Camp_Attended_df[Second_Health_Camp_Attended_df['Patient_ID'].isin(patients_visted_firstcamp)]['Patient_ID'].nunique()
first_second_visit = Second_Health_Camp_Attended_df[Second_Health_Camp_Attended_df['Patient_ID'].isin(patients_visted_firstcamp)]['Patient_ID'].unique()
print("Patients who have visited first and second camp ",n_first_second_visit)
first_camp_df=Patient_Profile_df.merge(First_Health_Camp_Attended_df,on='Patient_ID')
second_camp_df=Patient_Profile_df.merge(Second_Health_Camp_Attended_df,on='Patient_ID')
first_camp_df['First_camp']=1
second_camp_df['Second_camp']=2
first_merge_second = first_camp_df.merge(second_camp_df, on = 'Patient_ID')

