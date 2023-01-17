# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style="whitegrid")



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv('../input/noshowappointments/KaggleV2-May-2016.csv') ## read data 

df.head() ## explor data
df.shape 
df.info()
df["ScheduledDay"]= pd.to_datetime(df["ScheduledDay"]) 

df["AppointmentDay"]= pd.to_datetime(df["AppointmentDay"])  # covert to date object

df["AppointmentDay"].dtypes,df["ScheduledDay"].dtypes #check data type
df['diff_days']=df["AppointmentDay"]-df["ScheduledDay"] # caculate the period between scheduled day and appointment day 

df['diff_days'].value_counts()
df.head(10)
df_temp1=df[df.diff_days >= pd.to_timedelta(0)]

df_temp1  # remove minus period                #   df_temp1 to use in Q1
df_temp1['diff_days']=df_temp1['diff_days'].dt.days  # convert period to days
df_temp1['diff_days'].unique()
df_temp2=df_temp1.groupby(['diff_days','No-show'])['PatientId'].count().reset_index()

sns.set_context("notebook", font_scale=1.1)

sns.set_style("ticks")

f, ax = plt.subplots(figsize=(10, 30))

df_temp2=df_temp1.groupby(['diff_days','No-show'])['PatientId'].count().reset_index()



ax = sns.scatterplot(x="diff_days",y="PatientId", hue="No-show", data=df_temp2)

plt.title('Number of days for waiting and attending the appiontment')



# Set x-axis label

plt.xlabel('Period in days')



# Set y-axis label

plt.ylabel('Number of Patient')
df_temp3=df_temp2[(df_temp2['No-show']=='No')]   # for just who attend

f, ax = plt.subplots(figsize=(10, 30))

ax = sns.scatterplot(x="diff_days",y="PatientId", data=df_temp3)

plt.title('Number of days for waiting and attending the appiontment')



# Set x-axis label

plt.xlabel('Period')



# Set y-axis label

plt.ylabel('Number of Patient')
sns.distplot(df_temp2['diff_days']);

plt.xlabel('Period')
df.age.value_counts()
def age_stage(x):                              # Divide the age into stages

    if ((x>0 ) and (x<16)): x= "Child"

    elif((x>=16) and (x<30)): x="age from 16 to 29"

    elif((x>=30) and (x<40)): x="age from 30 to 39"

    elif((x>=40) and (x<50)): x="age from 40 to 49"

    elif((x>=50) and (x<60)): x="age from 50 to 59"

    elif((x>=60) and (x<70)): x="age from 60 to 69"

    elif((x>=70) and (x<80)): x="age from 70 to 79"    

    else: x="age from 79 to 100"

    return x;

        

df['age_stage']=df['Age'].apply(lambda x : age_stage(x))

df.head(5)
df_temp4=df.groupby(['age_stage','No-show'])['PatientId'].count().reset_index()  #df_temp4 for Q2

df_temp4.head()
sns.set(style="whitegrid")

f, ax = plt.subplots(figsize=(20, 10))

sns.barplot(x="age_stage", y="PatientId", hue="No-show",ax=ax,data=df_temp4,dodge=False);

plt.title('Age stages and number of Patient regards attend the appointments')



# Set x-axis label

plt.xlabel(' Age stage')



# Set y-axis label

plt.ylabel('Number of Patient')
df_temp5=df.groupby('No-show',as_index=False)['Scholarship','Hipertension','Diabetes','Alcoholism','Handcap','SMS_received'].sum()# for Q3
df_temp5.head()
df_temp=df_temp5.T.reset_index()          # make the coulmns as rows to compare between many variables 

df_temp.columns=['index','No','Yes']



df_temp
df_temp.drop(index=0,inplace=True) # remove No-show row

df_temp.head()


f, ax = plt.subplots(figsize=(20, 15))

labels=df_temp['index'].values

Y=df_temp['Yes'].values

X=np.arange(len(labels))

width=0.4



Yes=ax.barh(X,Y,width,label='not Attend')



######

Y=df_temp['No'].values

No=ax.barh((X+width),Y,width,label='Attend')







ax.set_xlabel('Number of Patient')

ax.set_title('Attend and many Variable')

ax.set_yticks(X)

ax.set_yticklabels(labels)

ax.legend(loc='best', fontsize=10)


