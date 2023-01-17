import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O 



import seaborn as sns

from matplotlib import pyplot as plt



import os

print(os.listdir("../input"))
df=pd.read_excel('../input/Hausaufgabe Data Science.xlsx',sheet_name='DataSample')
df.head()
df.dtypes
df.info()
#User_LTD represent the number of day since the user install the application.. So USER_LTD=0: the day of installation

df1 = df[df['USER_LTD']==0]
df1 = df1.groupby("VERSION_DATE_JOINED")['USER_ACTIVITY'].sum()

# Set the width and height of the figure

plt.figure(figsize=(10,6))



# Add title

plt.title("Nombre of new users, by version")



# Bar chart showing for each version how many new users have installed the application 

sns.barplot(x=df1.index, y=df1)



# Add label for vertical axis

plt.ylabel("# new user/version")
import time

from datetime import date

import datetime

datetime.datetime.today().weekday()
df1 = df[df['USER_LTD']==0]

df1=df1[['DATE_JOINED']]
day_list=list(df['DATE_JOINED'].unique())

dates = pd.DatetimeIndex(day_list)
day_number=[0,0,0,0,0,0,0]
for i in range(len(dates)):

    j=date(dates[i].year,dates[i].month, dates[i].day).weekday()

    day_number[j]=day_number[j]+1
day_number= pd.Series(day_number, index=[0, 1, 2, 3,4,5,6])
day_number.head()
#Return the day of the week as an integer, where Monday is 0 and Sunday is 6

df1['day']=df1['DATE_JOINED'].apply(lambda x:date(x.year,x.month, x.day).weekday())
df1 = df1.groupby("day").count()['DATE_JOINED']
df1.head()
df1=df1.divide(day_number)
df1.head()
# Set the width and height of the figure

plt.figure(figsize=(10,6))



# Add title

plt.title("The average of new users, by day")



sns.barplot(x=df1.index, y=df1)



# Add label for vertical axis

plt.ylabel("average of new user/day")
df1 = df[df['USER_ACTIVITY']==1]
#Return the day of the week as an integer, where Monday is 0 and Sunday is 6

df1['day']=df1['DATE_ACTIVITY'].apply(lambda x:date(x.year,x.month, x.day).weekday())
df1 = df1.groupby("day").count()['DATE_ACTIVITY']
day_number
df1=df1.divide(day_number)
import seaborn as sns

from matplotlib import pyplot as plt
# Set the width and height of the figure

plt.figure(figsize=(10,6))



# Add title

plt.title("In which day, they use the application the most")



sns.barplot(x=df1.index, y=df1)



# Add label for vertical axis

plt.ylabel("nomber of uses/day")
version=list(df['VERSION_DATE_ACTIVITY'].unique())
activ_version=[]

inactiv_version=[]

pourcentage_version=[]



for i in range(len(version)):

    df1=df[df['VERSION_DATE_ACTIVITY']==version[i]]

    activ_version.append(df1[df1['USER_ACTIVITY']==1].shape[0])

    inactiv_version.append(df1[df1['USER_ACTIVITY']==0].shape[0])

    pourcentage_version.append(activ_version[-1]*100 /(activ_version[-1]+inactiv_version[-1]))
df1=pd.DataFrame(data={'Version': version, '# of active user/version': activ_version,'# of inactive user/version': inactiv_version,'pourcentage_version':pourcentage_version})
df1.head()
# Set the width and height of the figure

plt.figure(figsize=(10,6))



# Add title

plt.title("% of uses per version")



sns.barplot(x=df1['Version'], y=df1['pourcentage_version'])



# Add label for vertical axis

plt.ylabel("% of use/day")
df.head()
df1 = df.groupby(['VERSION_DATE_JOINED','USER_LTD'])

df2 = df1.agg({'USER_ACTIVITY': np.mean})
df2=df2['USER_ACTIVITY'].unstack(0)

df2.head()
df2[['Version 1.3','Version 1.4','Version 1.4.1','Version 1.5','Version 1.6']].plot(figsize=(10,5))

plt.title('Cohorts: User Retention')

plt.xticks(np.arange(1, 154.1, 1))

plt.xlim(1, 154)

plt.ylabel('% active users');
days=[1,7,30,90]

percentage_actif_day=[]

for i in range(len(days)):

    df1=df[df['USER_LTD']==days[i]]

    percentage_actif_day.append(df1[df1['USER_ACTIVITY']==1].shape[0]*100/df1.shape[0])
df1=pd.DataFrame(data={'percentage of actif users':percentage_actif_day},index=days)
df1.head()
# Set the width and height of the figure

plt.figure(figsize=(10,6))



# Add title

plt.title("% of user that still active in day 1/7/30/90")



sns.barplot(x=df1.index, y=df1['percentage of actif users'])



# Add label for vertical axis

plt.ylabel("% of use/day")
df2.loc[[1]]
df2.loc[[7]]
df2.loc[[30]]
df2.loc[[90]]