# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt 
from datetime import datetime
import seaborn as sns
%matplotlib inline


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
# Read dataset
f = pd.read_csv('../input/KaggleV2-May-2016.csv')
df = pd.DataFrame(f)
## Manipulate the data to can use it in analysis process.

df.rename(columns={"No-show":"No_show"},inplace=True) ## rename column name
df.drop(df[df.Handcap.isin([2,3,4])].index,inplace=True) ## drop value doesn't meaningful to our analysis.
df = df[(df['Age'] < 100) & (df['Age'] > 0)] ## drop all ages more than 100 and less than 0 

## convert no-show status to numbers no-show =0 , show =1
df['No_show']=df['No_show'].str.replace('No' , '1')
df['No_show']=df['No_show'].str.replace('Yes' , '0')
df['No_show']=df['No_show'].astype(int)


### calculate waiting_time

df['ScheduledDay'] = df['ScheduledDay'].apply(lambda x: datetime.strptime(x, "%Y-%m-%dT%H:%M:%SZ").date())
df['AppointmentDay'] = df['AppointmentDay'].apply(lambda x: datetime.strptime(x, "%Y-%m-%dT%H:%M:%SZ").date())
df['Waiting_time'] = (df['AppointmentDay']-df['ScheduledDay']).astype('timedelta64[D]')

total_of_patinet=len(df)
df.head()
def calculate_percentage(name,df):
    """
    This fucntion will calculate all percentages of 'Scholarship','Hipertension',
    'Diabetes','Alcoholism','Handcap','SMS_received','No-show'for all patinets  (males & Females) 
    """
    
    if name == 'No_show':
        yes, no = df[name].value_counts()
        print('percentage of patinets no showe: {}%'.format(no/total_of_patinet*100))
        print('percentage of patinets Show up : {}%'.format(yes/total_of_patinet*100))
    else:
        not_have, have = df[name].value_counts()
        print('percentage of patinets not have {} : {}%'.format(name,not_have/total_of_patinet*100))
        print('percentage of patinets have {} : {}%'.format(name,have/total_of_patinet*100))

    
columns=['Scholarship','Hipertension','Diabetes','Alcoholism','Handcap','SMS_received','No_show']
for name in columns:
    calculate_percentage(name,df)
def calculate_percentage_Fmales(name,df_f):
    """
    This fucntion will calculate all percentages of 'Scholarship','Hipertension',
    'Diabetes','Alcoholism','Handcap','SMS_received','No-show'for all femals only 
    """    
    if name == 'No_show':
        yes, no = df_f[name].value_counts()
        no_percentage = no/total_of_patinet*100
        yes_percentage = yes/total_of_patinet*100
        
        print('percentage of female No Show: {}%'.format(no_percentage))
        print('percentage of female Show up : {}%'.format(yes_percentage))
        return no,yes
    else:
        not_have, have = df_f[name].value_counts()
        not_have_percentage = not_have/total_of_patinet*100
        yes_have_percentage = have/total_of_patinet*100
        
        print('percentage of female not have {} : {}%'.format(name,not_have_percentage))
        print('percentage of female have {} : {}%'.format(name,yes_have_percentage))
        return not_have,have

        
columns=['Scholarship','Hipertension','Diabetes','Alcoholism','Handcap','SMS_received','No_show']
df_f = df[df['Gender'] == 'F']
for name in columns:
    no, yes = calculate_percentage_Fmales(name,df_f)
# 5- what is the Neighbourhood which receiving high number of all patients got Scholarship.<br> 

data = df[['Neighbourhood','Scholarship']]
Neighbourhood_group = data.groupby(['Neighbourhood'])
total_results = Neighbourhood_group.sum()
total_results.sort_values('Scholarship',ascending=False)

total_results.plot(kind='bar',figsize=(20, 10),title='which Neighbourhood high number of all patients got Scholarship')
plt.ylabel("Scholarship")
plt.xlabel("Neighbourhood")

# max_neigh = total_results['Scholarship'].max()
# Neighbourhood=max_neigh[max_neigh['Scholarship'] == max_neigh].item()

# print("the Neighbourhood which receiving high number of all patients got Scholarship is {}".format(Neighbourhood))
# 6- which gender has more Scholarship <br> the asnwer is females
data = df[['Gender','Scholarship']]
gender_group = data.groupby(['Gender'])
total_gender = gender_group.sum()
total_gender.plot(kind='bar',title="which gender has more Scholarship")
plt.ylabel('Scholarship')
# 7-what is the patitent age has high number of show up. and they are females or males<br>
no_show_data =df[['No_show','Age','Gender']]
Age_df=no_show_data.groupby(['Age']).sum()
Age_df.sort_values('No_show',ascending=False,inplace=True)
Age_df.plot(kind='bar',figsize=(20,10),stacked=True,title='what is the patitent age has high number of show up')
#7.1 and they are females or males
# Age_df.drop('Age',axis=1,inplace=True)
data=df[['Gender','No_show']]
Age_df=data.groupby(['Gender']).sum()
Age_df.sort_values('No_show',ascending=False,inplace=True)
Age_df.plot(kind='bar',stacked=True,title='what is the gender has high number of show up')
plt.ylabel('No_show')
# 8- what is the patinet age has high number of patients are alcoholism.<br>
data=df[['Age','Alcoholism']]
alcoh_df=data.groupby('Age').sum()
alcoh_df.sort_values('Alcoholism',ascending=False,inplace=True)
alcoh_df.plot(kind='bar',figsize=(20,10),title='figure show the relation between patient age and Alcoholism')
plt.ylabel('Alcoholism')
# 9- which Neighbourhood receiving high percentage of patient show up?   the answer is JARDIM CAMBURI  
data = df[['Neighbourhood','No_show']]
neig_df=data.groupby('Neighbourhood').sum()
neig_df.plot(kind='bar',figsize=(20,10),title='graph of showing which neighbourhoods receiving high percentage of patient show up')
plt.ylabel('No_show')
# neig_df=data.groupby('Neighbourhood').sum()/total_of_patinet*100
# neig_df.sort_values('No_show',ascending=False,inplace=True)
## the percentage of patients had received high number SMS and show up associated with patient gender 
data = df[['No_show','SMS_received','Gender']]

age_df=data.groupby('Gender').sum()

# age_df.sort_values('No_show',ascending=False,inplace=True)
age_df.plot(kind='bar',figsize=(20,10),stacked=True,title="which patient gender has highing number Received SMS ans show up")
plt.ylabel('SMS_received / No_show')

age_df['Ratio_of_SMS_received']= age_df['SMS_received']/total_of_patinet

age = (age_df['SMS_received']/total_of_patinet*100).max()
age
data = df[['No_show','Scholarship','SMS_received']]
Scholarship_show = data.groupby('Scholarship').sum()
Scholarship_show.plot(kind='bar')
s_df = df[['No_show','Waiting_time']]

result_s_df = s_df.groupby(['No_show']).mean()
result_s_df.rename(columns={'Waiting_time':'Avg_waiting_time'},inplace=True)
result_s_df.sort_values('Avg_waiting_time', ascending=False, inplace=True)
result_s_df.plot(kind='bar',figsize=(10,5),stacked=True,title='Average waiting time for patient showed up')

