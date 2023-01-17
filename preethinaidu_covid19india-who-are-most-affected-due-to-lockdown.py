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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
df = pd.read_csv('../input/covid-19-india/covidindia.csv')
df.head()
df.shape
df.groupby(['detected_state','gender'])[['gender']].size().unstack().plot(kind='barh',figsize=(15,10))
df['age'].unique()
df.loc[df['age']== -1]
df.drop(df.loc[df['age']== -1].index , inplace = True)
age_10 = df[df['age'].between(0,10,inclusive = True)]['age'].count()
print('cases less than 10 age group' , age_10)
age_20 = df[df['age'].between(11,20,inclusive = True)]['age'].count()
print('__________________')
print('cases less than 20 age group' , age_20)
age_30 = df[df['age'].between(21,30,inclusive = True)]['age'].count()
print('__________________')
print('cases less than 30 age group' , age_30)
age_40 = df[df['age'].between(31,40,inclusive = True)]['age'].count()
print('__________________')
print('cases less than 40 age group' , age_40)
age_50 = df[df['age'].between(41,50,inclusive = True)]['age'].count()
print('__________________')
print('cases less than 50 age group' , age_50)
age_60 = df[df['age'].between(51,60,inclusive = True)]['age'].count()
print('__________________')
print('cases less than 60 age group' , age_60)
age_70 = df[df['age'].between(61,70,inclusive = True)]['age'].count()
print('__________________')
print('cases less than 70 age group' , age_70)
age_80 = df[df['age'].between(71,80,inclusive = True)]['age'].count()
print('__________________')
print('cases less than 80 age group' , age_80)
age_90 = df[df['age'].between(81,90,inclusive = True)]['age'].count()
print('__________________')
print('cases less than 90 age group' , age_90)
age_100 = df[df['age'].between(91,100,inclusive = True)]['age'].count()
print('__________________')
print('cases less than 1000 age group' , age_100)
age_grp = pd.DataFrame([14,23,129,94,66,74,58,9,4,1],
                       index = ['0-10','10-20','20-30','30-40','40-50','50-60','60-70','70-80','80-90','90-100'],
                      columns = ['cases'])
age_grp
age_grp.plot(kind='pie',figsize=(8,8),subplots='True')
cor = df.corr()
sns.heatmap(cor,annot = True)
df.drop(columns=['unique_id','contacts'],axis = 1,inplace = True)
df.head()
df['current_status'].value_counts()
Status = pd.DataFrame([974,31,18],index = ['Hospitalized','Recovered','Deceased'],columns = ['Count'])
Status
Status.plot(kind='pie',subplots=True,figsize=(6,6))
df['diagnosed_date'].unique()
Education_sector = (df['diagnosed_date'] > '2020-01-30') & (df['diagnosed_date'] <= '2020-03-08')
Education_sector = len(df[Education_sector])
Public_place = (df['diagnosed_date'] > '2020-01-30') & (df['diagnosed_date'] <= '2020-03-12')
Public_place = len(df[Public_place])
Work_from_home = (df['diagnosed_date'] > '2020-01-30') & (df['diagnosed_date'] <= '2020-03-15')
Work_from_home = len(df[Work_from_home])
Lockdown = (df['diagnosed_date'] > '2020-01-30') & (df['diagnosed_date'] <= '2020-03-28')
Lockdown = len(df[Lockdown])
Sectors_count = [Education_sector,Public_place,Work_from_home,Lockdown]
dates = ['2020-03-08','2020-03-12','2020-03-15','2020-03-28']
df2 = pd.DataFrame({'Sectors_count':Sectors_count,'labels':['schools shutdown','public places shutdown',
            'work from home started','country under lockdown'],'dates':dates})
df2
df2.groupby(['dates','labels','Sectors_count'])[['Sectors_count','labels']].size().unstack().plot(kind='bar')
