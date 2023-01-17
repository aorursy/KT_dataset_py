"""
Author: Carlos Abiera
Date: 23 October 2018

"""
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import os
#print(os.listdir("../input"))
df = pd.read_csv("../input/att.csv") #read file
# display top rows using head 
df.head()
# print summary statistics
df.describe()
df.drop(['attn_nID','empl_nEmployeeID','attn_nBadgeID','dvce_nID','offc_nOfficeID','Expr1'],axis=1, inplace=True)
df.head() #O = logout while I = login, 
df = df[df['attn_sLogType'] == 'I'] #select only all IN
df.head() #Let's check
#CHECK ALL DATE DUPLICATE
dt = df['attn_dDateTime']
checkdup = df[df.duplicated(keep=False)]
checkdup

#REMOVE ALL DUPLICATE BY DATE AND LOGIN CRITERIA
df = df.drop_duplicates(['attn_dDateTime', 'attn_sLogType'])
df[df.duplicated(keep=False)] #check again
df['attn_sLogType'].isnull().sum()
df['attn_dDateTime'].isnull().sum()
d = pd.to_datetime(df['attn_dDateTime']) #convert column to datetime data type
df['ByDate'] = d.dt.date #assign converted value to row
df['ByMonth'] = d.dt.month  #Jan=1, Feb=2 , ..
df['ByDayWeek'] = d.dt.weekday_name  #Monday, Tuesday, WW
df['ByDayWeekNum'] = d.dt.dayofweek  #Monday=0, Sunday=6
df['ByTime'] = d.dt.time #TIME
df['ByYear'] = d.dt.year #YEAR
df.head() #CHECK 
#REMOVE ALL LOGIN MORE THAN 10AM
ch = pd.to_datetime('10:00:00',format= '%H:%M:%S' )
df = df[((df['ByTime']) < (ch.time())) & (df['ByDayWeekNum'] != 5) & (df['ByDayWeekNum'] != 6) ]
df.head() #LEZ CHECK

#LABEL LATE OR NOT
att = pd.to_datetime('09:05:00',format= '%H:%M:%S' )
df['Late']=np.where(df['ByTime'] > att.time(), 1, 0)
df.head() #LEZ CHECK
df.dtypes #CHECK COLUMN NAME AND TYPE
#df['Late'].value_counts().plot.bar()
#data['Status'].value_counts().plot.bar()
df_late = df.loc[df['Late']==1] # show late only
df_late['ByYear'].value_counts()
df_late['ByYear'].value_counts().plot.bar()
df_late_2018 = df_late[df['ByYear']==2018]
df_late_2018['ByMonth'].value_counts().sort_values(ascending=False).sort_index()

df_late_2018['ByMonth'].value_counts().sort_values(ascending=False).sort_index().plot.bar(title="2018")
df_late['ByMonth'].value_counts().sort_values(ascending=False).sort_index().plot.bar(title="2012-2018")

pd.Categorical(df_late['ByDayWeek'], categories=['Monday','Tuesday','Wednesday','Thursday','Friday'], ordered=True).value_counts().sort_values(ascending=False).sort_index().plot.bar(title="2012-2018")
import seaborn as sns
sns.set(style="white") #white background style for seaborn plots
sns.set(style="whitegrid", color_codes=True)

plt.figure(figsize=(15,8))
ax = sns.kdeplot(df["ByDayWeekNum"][df.Late == 1], color="darkturquoise", shade=True)
sns.kdeplot(df["ByDayWeekNum"][df.Late == 0], color="lightcoral", shade=True)

plt.legend(['Late', 'Not Late'])
plt.title('Density Plot of Number of Late ')
ax.set(xlabel='Day')
#plt.xlim(-10,85)
plt.show()
