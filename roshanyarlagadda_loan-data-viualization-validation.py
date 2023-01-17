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
df=pd.read_csv("/kaggle/input/loandata/Loan payments data.csv")
df.head()
df.tail(10)
df.describe()
df.info()
df.isnull().sum()
df['age'].describe()
print(df[['age','terms']].describe())
df.corr()
df.corr()
df.dtypes
df.size
print("Data columns\n")

for col in df.columns:

    print(col)
print("Rename data columns\n")

df=df.rename(columns={'Loan_ID':'Loan_Id','loan_status':'Loan_Status','terms':'Terms','effective_date':'Effective_Date',

                     'due_date':'Due_Date','paid_off_time':'Paid_Off_Time','past_due_days':'Past_Due_Days','age':'Age',

                     'education':'Education'})



for col in df.columns:

    print(col)
print("Loan_Id values\n")

print(df.Loan_Id.unique())
print("types of features in Loan_Status")

print(df.Loan_Status.unique())

loan_status=df.Loan_Status.unique().reshape(-1,1)

print(loan_status)
paidoff_count=0

collection_count=0

collection_paidoff_count=0



print('Loan status with Loan_Id counts\n')

paidoff_count=len(df[df.Loan_Status==loan_status[0][0]].Loan_Id)

collection_count=len(df[df.Loan_Status==loan_status[1][0]].Loan_Id)

collection_paidoff_count=len(df[df.Loan_Status==loan_status[2][0]].Loan_Id)  

print(paidoff_count,collection_count,collection_paidoff_count)
df.Principal.unique

print("Principle coloumn\n")

print(df.Principal.value_counts())
print("Loan_Status and Principle")

print(df.groupby('Loan_Status')['Principal'].value_counts())
print("Total amount")

print(df.groupby('Loan_Status')['Principal'].sum())
print("Unique Terms\n")

print(df.Terms.value_counts())

print(df.Terms.unique())
print("relation of terms and principal\n")

print(df.groupby('Terms')['Principal'].value_counts())

print("Groupby sum of terms and principal\n")

print(df.groupby('Terms')['Principal'].sum())
print("effective date\n")

print(df.Effective_Date.isnull().sum())

print("features of effective date\n")

print(df.Effective_Date.unique())

print(df.Effective_Date.value_counts())

print("Group by effective date and terms and principal")

print(df.groupby('Effective_Date')['Terms','Principal'].count())

print(df.groupby('Effective_Date')['Terms','Principal'].sum())

print(df.groupby('Effective_Date')['Terms','Principal'].mean())
print('Age in Data state\n')

print('Data isnull sum\n')

print(df.Age.isnull().sum())

print('Data unique age\n')

print(df.Age.unique())

print('Data value counts age\n')

print(df.Age.value_counts())

print('Age group by Terms nad Principal in every Data\n')

print(df.groupby('Terms')['Age'].value_counts())

print('Group sum every Terms and Principal in Data')

print(df.groupby('Age')[['Terms','Principal']].sum())
df_age=df.Age.value_counts().index

len(df_age)

count=0

df_age_list=[]

for age in df_age:

    df_age_list.append(sum(df[df.Age==int(age)].Principal))

    print(df_age_list[count],age)

    count=count+1
print("Education\n")

print(df.Education.unique())

print(df.Education.value_counts())

print(df.groupby(['Education','Terms'])['Principal'].sum())

print(df.groupby('Education')['Age'].count())
print("Unique values of Gender\n")

print(df.Gender.unique())

print("null values\n")

print("null values in Gender column ",df.Gender.isnull().sum())

print("Gender and Principal\n")

print(df.groupby('Gender')['Principal'].sum())

print(df.groupby(['Gender','Terms'])['Principal'].sum())

print("Gender , Age , Principal")

print(df.groupby(['Gender','Age'])['Principal'].count())

print(df.groupby(['Gender','Age'])['Principal'].sum())
print("due_date unique values\n")

print(df.Due_Date.unique())

print("paid_off unique values\n")

print(df.Paid_Off_Time.unique())

print("null values of due_date and paid_off_time\n")

print(df.Due_Date.isnull().sum())

print(df.Paid_Off_Time.isnull().sum())

print("Due_date vs Paid_Off_Time count\n")

print(df.groupby(['Due_Date','Paid_Off_Time']).count())

print("Due_date vs Paid_Off_Time sum\n")

print(df.groupby(['Due_Date','Paid_Off_Time']).sum())
print("past_due_days\n")

print(df.Past_Due_Days.unique())

print("null values of past_due_days\n")

print(df.Past_Due_Days.isnull().sum())

print("past_due_days value counts")

print(df.Past_Due_Days.value_counts())
#Data Cleaning



df=df.drop(columns=['Loan_Id'],axis=1)

print(df)
df.corr()
df=df.drop(['Past_Due_Days','Paid_Off_Time'],axis=1)

print(df)
import seaborn as sns

import matplotlib.pyplot as plt

sns.barplot(x=df[df['Loan_Status']=='PAIDOFF'].groupby('Education')['Gender'].count().index,

            y=df[df['Loan_Status']=='PAIDOFF'].groupby('Education')['Gender'].count().values)

plt.xticks(rotation=90)

plt.show()
from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

df['Loan_Status']=le.fit_transform(df['Loan_Status'])
df[df['Principal']==1000].Gender.value_counts()
plt.figure(figsize=(5,5))

ax=sns.barplot(x=df[df['Principal']==1000].Gender.value_counts().index,

              y=df[df['Principal']==1000].Gender.value_counts().values,

              palette=sns.cubehelix_palette(120))

plt.xlabel('Principal')

plt.ylabel('Gender')

plt.title('Show Principal & Gender Bar Plot')

plt.show()
le=LabelEncoder()

df['Principal']=le.fit_transform(df['Principal'])
labels=df['Terms'].value_counts().index

colors=['blue','red','yellow']

explode=[0,0,0.1]

values=df['Terms'].value_counts().values



#visualization

plt.figure(figsize=(7,7))

plt.pie(values,explode=explode,labels=labels,colors=colors,autopct='%1.1f%%')

plt.title('Ters According Analysis',color='black',fontsize=10)

plt.show()